"""Core unit tests for Kalman filtering engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.kalman import (
    KalmanFilterModel,
    KalmanRegimeModel,
    KalmanSettings,
    LinearGaussianSSM,
    build_system,
    filter_adaptive,
    filter_ekf,
    filter_linear,
    filter_ukf,
    rts_smooth,
    sigma_points,
    simulate_lds,
    unscented_transform,
)
from iqrp.app.regimes.kalman.covariance import block_diag, ensure_spd, joseph_update, mahalanobis
from iqrp.app.regimes.kalman.initialization import numerical_jacobian
from iqrp.app.regimes.kalman.prediction import (
    forecast_observation,
    n_step_predict,
    prediction_intervals,
    predict_nonlinear,
    predict_state,
)
from iqrp.app.regimes.kalman.update import innovation_statistics, update_nonlinear, update_state
from iqrp.app.regimes.kalman.visualization import (
    plot_covariance_evolution,
    plot_filtered_state,
    plot_innovations,
    plot_kalman_gain,
    plot_prediction_bands,
    plot_smoothed_state,
)


def _settings(**kw: object) -> KalmanSettings:
    data = {**KalmanSettings.default().model_dump(), "training": {"em_iterations": 3, "tol": 1e-3, "estimate_noise": True}}
    data.update(kw)
    return KalmanSettings.from_mapping(data)


def _lds(seed: int = 0, n: int = 80) -> tuple[LinearGaussianSSM, np.ndarray, np.ndarray]:
    settings = _settings(application="trend", filter_type="linear")
    sys = build_system(settings, application="trend")
    states, obs = simulate_lds(sys, n, rng=np.random.default_rng(seed))
    return sys, states, obs


@pytest.mark.unit
def test_covariance_and_predict_update() -> None:
    p = ensure_spd([[2.0, 0.5], [0.5, 1.0]])
    assert p.shape == (2, 2)
    assert block_diag(np.eye(1), np.eye(2)).shape == (3, 3)
    x, p2 = predict_state(np.zeros(2), p, np.eye(2), 0.1 * np.eye(2))
    assert x.shape == (2,)
    x3, p3, innov, s, k = update_state(x, p2, np.array([0.1]), np.array([[1.0, 0.0]]), np.array([[0.1]]))
    assert innov.shape[0] == 1 and k.shape == (2, 1)
    assert joseph_update(p2, k, np.array([[1.0, 0.0]]), np.array([[0.1]])).shape == (2, 2)
    assert mahalanobis(innov, s) >= 0
    lo, hi = prediction_intervals(x3, p3)
    assert lo.shape == hi.shape
    means, covs = n_step_predict(x3, p3, np.eye(2), 0.1 * np.eye(2), horizon=3)
    assert means.shape == (3, 2) and covs.shape == (3, 2, 2)
    yhat, s2 = forecast_observation(x3, p3, np.array([[1.0, 0.0]]), np.array([[0.1]]))
    assert yhat.shape[0] == 1 and s2.shape == (1, 1)


@pytest.mark.unit
def test_linear_filter_and_rts() -> None:
    sys, states, obs = _lds(1)
    trace = filter_linear(obs, sys)
    assert trace.means.shape[0] == obs.shape[0]
    assert np.isfinite(trace.log_likelihood)
    sm = rts_smooth(trace, sys)
    # smoothed should be closer to truth on average than filtered for mid points
    filt_mse = float(np.mean((trace.means - states) ** 2))
    sm_mse = float(np.mean((sm.means - states) ** 2))
    assert sm_mse <= filt_mse * 1.05 + 0.05
    stats = innovation_statistics(trace.innovations, trace.innovation_covs)
    assert stats["n"] == float(obs.shape[0])


@pytest.mark.unit
def test_ekf_ukf_adaptive_and_jacobians() -> None:
    settings = _settings(application="volatility", filter_type="ekf")
    sys = build_system(settings, application="volatility")
    _, obs = simulate_lds(sys, 60, rng=np.random.default_rng(2))
    # use positive proxies
    obs = np.abs(obs) + 0.01
    ekf = filter_ekf(obs, sys)
    ukf = filter_ukf(obs, sys, alpha=1e-2, beta=2.0, kappa=0.0)
    assert np.isfinite(ekf.log_likelihood) and np.isfinite(ukf.log_likelihood)
    pts, wm, wc = sigma_points(sys.x0, sys.p0, alpha=1e-2)
    mean, cov = unscented_transform(pts, wm, wc, sys.q)
    assert mean.shape[0] == 1 and cov.shape == (1, 1)
    denoise = build_system(_settings(application="denoise"), application="denoise")
    _, y = simulate_lds(denoise, 50, rng=np.random.default_rng(3))
    adapt = filter_adaptive(y, denoise, window=10)
    assert "q_final" in adapt.metadata

    def f(x: np.ndarray) -> np.ndarray:
        return np.array([float(x[0]) ** 2])

    jac = numerical_jacobian(f, np.array([2.0]))
    assert jac.shape == (1, 1) and abs(float(jac[0, 0]) - 4.0) < 0.01
    xn, pn = predict_nonlinear(np.array([1.0]), np.eye(1), f, lambda x: numerical_jacobian(f, x), np.eye(1) * 0.01)
    assert xn.shape[0] == 1
    xu, pu, *_ = update_nonlinear(
        xn, pn, np.array([1.0]), f, lambda x: numerical_jacobian(f, x), np.array([[0.1]])
    )
    assert xu.shape[0] == 1


@pytest.mark.unit
def test_model_api(tmp_path: Path) -> None:
    sys, states, obs = _lds(4, n=100)
    model = KalmanFilterModel(settings=_settings(application="trend"), random_seed=4, system=sys)
    model.fit(obs)
    pred = model.predict(obs)
    proba = model.predict_proba(obs)
    assert pred.shape[0] == obs.shape[0]
    assert proba.shape == (obs.shape[0], 2)
    assert np.isfinite(model.score(obs))
    sm = model.smooth(obs)
    assert sm.smoothed_states.shape[0] == obs.shape[0]
    fc = model.forecast(obs, horizon=4)
    assert fc.horizon == 4
    labels, samp = model.sample(30)
    assert labels.shape[0] == 30 and samp.shape[0] == 30
    assert model.state().shape[0] == 2
    assert model.covariance().shape == (2, 2)
    assert model.innovation().shape[0] >= 1
    assert model.kalman_gain().shape[0] == 2
    assert model.filtered_means().shape[0] == obs.shape[0]
    assert model.smoothed_means().shape[0] == obs.shape[0]
    x_new = model.update(obs[-1])
    assert x_new.shape[0] == 2
    path = model.save(tmp_path / "kf.json")
    loaded = KalmanFilterModel.load(path)
    assert loaded.is_fitted
    diag = model.diagnostics(obs)
    assert "filter_stability" in diag
    report = model.evaluate(obs, true_states=states)
    assert report["metrics"]["state_corr"] > 0.5
    plot_filtered_state(model.filtered_means(), tmp_path / "f.svg", observations=obs[:, 0])
    plot_smoothed_state(model.smoothed_means(), tmp_path / "s.svg")
    lo, hi = prediction_intervals(model.state(), model.covariance())
    plot_prediction_bands(model.filtered_means()[:, 0], lo[0] * np.ones(obs.shape[0]), hi[0] * np.ones(obs.shape[0]), tmp_path / "b.svg")
    plot_innovations(model._trace.innovations, tmp_path / "i.svg")  # type: ignore[union-attr]
    plot_covariance_evolution(model._trace.covs, tmp_path / "c.svg")  # type: ignore[union-attr]
    plot_kalman_gain(model._trace.gains, tmp_path / "g.svg")  # type: ignore[union-attr]
    frame = pl.DataFrame({"open_time": list(range(obs.shape[0])), "close": obs[:, 0]})
    regime = KalmanRegimeModel(settings=_settings(application="trend"), random_seed=5)
    regime.fit(frame, feature_columns=["close"])
    assert regime.predict_proba(frame, feature_columns=["close"]).shape[1] == 2
    assert regime.forecast(frame, steps=2).steps == 2


@pytest.mark.unit
def test_applications_and_config() -> None:
    s = KalmanSettings.from_hydra(overrides=["application=denoise", "filter_type=adaptive"])
    assert s.application == "denoise"
    for app in ("trend", "denoise", "dynamic_beta", "volatility", "spread", "pairs", "custom"):
        sys = build_system(_settings(application=app), application=app)  # type: ignore[arg-type]
        assert sys.n_states >= 1 and sys.n_obs >= 1
    bad = ensure_spd(np.array([[1.0]]))
    assert bad[0, 0] > 0
