"""Gap-filling tests for Kalman coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.kalman import KalmanFilterModel, KalmanSettings, build_system, simulate_lds
from iqrp.app.regimes.kalman.covariance import ensure_spd
from iqrp.app.regimes.kalman.ekf import filter_ekf
from iqrp.app.regimes.kalman.initialization import LinearGaussianSSM, numerical_jacobian
from iqrp.app.regimes.kalman.linear import filter_linear
from iqrp.app.regimes.kalman.prediction import predict_state
from iqrp.app.regimes.kalman.serializer import KalmanSerializer
from iqrp.app.regimes.kalman.ukf import filter_ukf
from iqrp.app.regimes.kalman.update import update_state
from iqrp.app.regimes.kalman.visualization import (
    plot_covariance_evolution,
    plot_filtered_state,
    plot_innovations,
    plot_kalman_gain,
    plot_prediction_bands,
    plot_smoothed_state,
)
from iqrp.app.state_space import get_registry as get_ss_registry
from iqrp.app.regimes.base.registry import get_registry as get_regime_registry

import iqrp.app.regimes.kalman  # noqa: F401


def _settings(**kw: object) -> KalmanSettings:
    data = {**KalmanSettings.default().model_dump(), "training": {"em_iterations": 2, "tol": 1e-3, "estimate_noise": False}}
    data.update(kw)
    return KalmanSettings.from_mapping(data)


@pytest.mark.unit
def test_registry_and_system_dict() -> None:
    assert "kalman" in get_ss_registry().list_names()
    assert "kalman" in get_regime_registry().list_names()
    sys = build_system(_settings(application="trend"), application="trend")
    d = sys.to_dict()
    sys2 = LinearGaussianSSM.from_dict(d)
    assert sys2.n_states == sys.n_states
    assert sys2.b is None


@pytest.mark.unit
def test_predict_with_control_and_pinv_paths() -> None:
    f = np.eye(2)
    b = np.array([[1.0], [0.0]])
    x, p = predict_state(np.zeros(2), np.eye(2), f, 0.01 * np.eye(2), b=b, u=np.array([2.0]))
    assert abs(float(x[0]) - 2.0) < 1e-9
    # near-singular S → pinv branch
    h = np.array([[1.0, 0.0]])
    r = np.array([[1e-30]])
    p_bad = np.array([[1e-30, 0.0], [0.0, 1e-30]])
    *_rest, k = update_state(np.zeros(2), p_bad, np.array([0.0]), h, r)
    assert k.shape == (2, 1)
    # ensure_spd edge cases
    assert ensure_spd(0.0).shape == (1, 1)
    assert ensure_spd(np.array([1.0, 2.0])).shape == (2, 2)
    # eigh failure simulation via non-finite? just large asymmetric
    m = ensure_spd(np.array([[1.0, 100.0], [0.0, 1.0]]))
    assert m.shape == (2, 2)


@pytest.mark.unit
def test_ekf_ukf_defaults_without_hooks() -> None:
    sys = build_system(_settings(application="denoise"), application="denoise")
    # strip hooks
    sys = LinearGaussianSSM(f=sys.f, h=sys.h, q=sys.q, r=sys.r, x0=sys.x0, p0=sys.p0)
    _, obs = simulate_lds(sys, 20, rng=np.random.default_rng(1))
    assert np.isfinite(filter_ekf(obs, sys).log_likelihood)
    assert np.isfinite(filter_ukf(obs.reshape(-1), sys).log_likelihood)


@pytest.mark.unit
def test_volatility_roundtrip_and_columns(tmp_path: Path) -> None:
    settings = _settings(application="volatility", filter_type="ekf")
    sys = build_system(settings, application="volatility")
    _, obs = simulate_lds(sys, 40, rng=np.random.default_rng(2))
    obs = np.abs(obs) + 0.05
    model = KalmanFilterModel(settings=settings, system=sys, random_seed=2)
    model.fit(obs)
    model.smooth(obs)
    path = KalmanSerializer().save(model, tmp_path / "vol.json")
    loaded = KalmanSerializer().load(path, model_cls=KalmanFilterModel)
    assert loaded.system is not None
    assert loaded.system.h_fn is not None
    # configured observation columns
    settings2 = _settings(
        application="denoise",
        columns={"timestamp": "open_time", "observation_columns": ["px"], "control_columns": ["u"]},
    )
    frame = pl.DataFrame({"open_time": [1, 2, 3], "px": [1.0, 1.1, 1.2], "u": [0.0, 0.1, 0.0]})
    m = KalmanFilterModel(settings=settings2)
    m.fit(frame)
    assert m.is_fitted


@pytest.mark.unit
def test_viz_empty_and_bands(tmp_path: Path) -> None:
    settings = _settings(visualization={"enabled": True, "max_points": 50})
    plot_filtered_state(np.array([]), tmp_path / "empty.svg", settings)
    plot_smoothed_state(np.linspace(0, 1, 10), tmp_path / "sm.svg", settings)
    plot_prediction_bands(np.zeros(10), -np.ones(10), np.ones(10), tmp_path / "pb.svg", settings)
    plot_innovations(np.random.default_rng(0).normal(size=(10, 2)), tmp_path / "inn.svg", settings)
    plot_covariance_evolution(np.stack([np.eye(2) for _ in range(10)]), tmp_path / "cov.svg", settings)
    plot_kalman_gain(np.stack([np.ones((2, 1)) for _ in range(10)]), tmp_path / "kg.svg", settings)


@pytest.mark.unit
def test_rts_with_f_seq_and_1d_adapt() -> None:
    sys = build_system(_settings(application="denoise"), application="denoise")
    _, obs = simulate_lds(sys, 15, rng=np.random.default_rng(3))
    tr = filter_linear(obs, sys)
    from iqrp.app.regimes.kalman.smoothing import rts_smooth
    from iqrp.app.regimes.kalman.adaptive import filter_adaptive, adapt_noise_from_trace

    sm = rts_smooth(tr, sys, f_seq=np.stack([sys.f] * 15))
    assert sm.means.shape[0] == 15
    # short series adaptive + 1d cov path
    tr2 = filter_adaptive(obs[:5].reshape(-1), sys, window=20)
    q, r = adapt_noise_from_trace(tr2, sys)
    assert q.shape[0] == 1
    # empty diff path
    tr3 = FilterTraceMini(tr2.means[:1])
    q2, _ = adapt_noise_from_trace_empty(tr3, sys)
    assert q2.shape == sys.q.shape


class FilterTraceMini:
    def __init__(self, means: np.ndarray) -> None:
        self.means = means
        self.innovations = np.array([[0.1]])


def adapt_noise_from_trace_empty(trace: FilterTraceMini, system: LinearGaussianSSM):
    from iqrp.app.regimes.kalman.adaptive import adapt_noise_from_trace
    from iqrp.app.regimes.kalman.linear import FilterTrace

    ft = FilterTrace(
        means=trace.means,
        covs=np.array([np.eye(1)]),
        pred_means=trace.means,
        pred_covs=np.array([np.eye(1)]),
        innovations=trace.innovations,
        innovation_covs=np.array([[[0.1]]]),
        gains=np.array([[[1.0]]]),
        log_likelihood=0.0,
    )
    return adapt_noise_from_trace(ft, system)


@pytest.mark.unit
def test_jacobian_and_numerical() -> None:
    def h(x: np.ndarray) -> np.ndarray:
        return np.array([np.sin(x[0]), x[0] * x[1] if x.size > 1 else x[0]])

    jac = numerical_jacobian(h, np.array([0.5, 1.0]))
    assert jac.shape[0] == 2
