"""Synthetic recovery, stability, and stress tests for Kalman engine."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.regimes.kalman import (
    KalmanFilterModel,
    KalmanSettings,
    build_system,
    filter_linear,
    rts_smooth,
    simulate_lds,
)
from iqrp.app.regimes.kalman.covariance import ensure_spd
from iqrp.app.simulation.stochastic.ou import OrnsteinUhlenbeck


def _settings(**kw: object) -> KalmanSettings:
    data = {
        **KalmanSettings.default().model_dump(),
        "training": {"em_iterations": 4, "tol": 1e-4, "estimate_noise": True},
    }
    data.update(kw)
    return KalmanSettings.from_mapping(data)


@pytest.mark.unit
def test_recover_latent_trend() -> None:
    settings = _settings(application="trend", filter_type="linear")
    sys = build_system(settings, application="trend")
    # lower observation noise for recovery
    sys = type(sys)(
        f=sys.f, h=sys.h, q=sys.q * 0.5, r=np.array([[1e-3]]), x0=sys.x0, p0=sys.p0, application="trend"
    )
    states, obs = simulate_lds(sys, 400, rng=np.random.default_rng(11))
    model = KalmanFilterModel(settings=settings, system=sys, random_seed=11)
    model.fit(obs)
    report = model.evaluate(obs, true_states=states)
    assert report["metrics"]["state_corr"] >= 0.85
    assert report["metrics"]["smooth_rmse"] <= report["metrics"]["state_rmse"] * 1.05 + 0.1
    assert np.isfinite(report["metrics"]["log_likelihood"])


@pytest.mark.unit
def test_simulation_engine_ou_denoise() -> None:
    ou = OrnsteinUhlenbeck(rng=np.random.default_rng(7))
    path = ou.generate(300, x0=100.0, dt=0.01, mean_reversion_speed=0.5, mean_reversion_level=100.0, volatility=2.0)
    prices = np.asarray(path.prices, dtype=np.float64).reshape(-1)
    # add sensor noise
    noisy = prices + np.random.default_rng(7).normal(0, 0.5, size=prices.size)
    settings = _settings(application="denoise", filter_type="linear")
    model = KalmanFilterModel(settings=settings, random_seed=7)
    model.fit(noisy)
    filt = model.filtered_means()[:, 0]
    # filtered closer to clean than noisy
    assert float(np.mean((filt - prices[: filt.size]) ** 2)) < float(np.mean((noisy[: filt.size] - prices[: filt.size]) ** 2))


@pytest.mark.unit
def test_numerical_stability_extreme() -> None:
    settings = _settings(application="custom", n_states=2, n_obs=1)
    sys = build_system(settings)
    sys = type(sys)(
        f=np.eye(2),
        h=np.array([[1.0, 0.0]]),
        q=1e-12 * np.eye(2),
        r=np.array([[1e-12]]),
        x0=np.array([1e6, 0.0]),
        p0=1e6 * np.eye(2),
        application="custom",
    )
    rng = np.random.default_rng(0)
    _, obs = simulate_lds(sys, 100, rng=rng)
    obs = obs * 1e6
    trace = filter_linear(obs, sys)
    assert np.all(np.isfinite(trace.means))
    assert np.all(np.isfinite(trace.covs))
    sm = rts_smooth(trace, sys)
    assert np.all(np.isfinite(sm.means))
    assert float(np.min([np.min(np.linalg.eigvalsh(ensure_spd(c))) for c in trace.covs])) > 0


@pytest.mark.unit
def test_stress_large_series() -> None:
    settings = _settings(application="denoise", filter_type="linear", training={"em_iterations": 1, "tol": 1e-2, "estimate_noise": False})
    sys = build_system(settings, application="denoise")
    n = 200_000
    states, obs = simulate_lds(sys, n, rng=np.random.default_rng(3))
    model = KalmanFilterModel(settings=settings, system=sys, random_seed=3)
    model.fit(obs)
    # subsample evaluate
    idx = slice(0, 5000)
    report = model.evaluate(obs[idx], true_states=states[idx])
    assert report["metrics"]["state_corr"] > 0.7
    assert np.isfinite(model.log_likelihood(obs[idx]))
