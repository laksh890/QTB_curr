"""Broad coverage tests for Kalman modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from iqrp.app.core.exceptions import ConfigurationError, ValidationError
from iqrp.app.regimes.kalman.adaptive import adapt_noise_from_trace, filter_adaptive
from iqrp.app.regimes.kalman.config import KalmanSettings
from iqrp.app.regimes.kalman.covariance import ensure_spd
from iqrp.app.regimes.kalman.diagnostics import KalmanDiagnostics, _lag1_corr
from iqrp.app.regimes.kalman.evaluator import KalmanEvaluator, stats_chi2_crit
from iqrp.app.regimes.kalman.initialization import LinearGaussianSSM, build_system
from iqrp.app.regimes.kalman.linear import FilterTrace, _gaussian_ll, filter_linear
from iqrp.app.regimes.kalman.model import KalmanFilterModel, KalmanRegimeModel, _soft_trend_proba
from iqrp.app.regimes.kalman.prediction import predict_state
from iqrp.app.regimes.kalman.serializer import KalmanSerializer, _json_default
from iqrp.app.regimes.kalman.smoothing import rts_smooth
from iqrp.app.regimes.kalman.trainer import KalmanTrainer, run_filter, simulate_lds
from iqrp.app.regimes.kalman.ukf import filter_ukf, sigma_points
from iqrp.app.regimes.kalman.visualization import _ensure, plot_filtered_state


def _settings(**kw: object) -> KalmanSettings:
    data = {
        **KalmanSettings.default().model_dump(),
        "training": {"em_iterations": 2, "tol": 1e-3, "estimate_noise": True},
    }
    data.update(kw)
    return KalmanSettings.from_mapping(data)


@pytest.mark.unit
def test_config_paths_and_errors() -> None:
    s = KalmanSettings.default()
    assert s.enabled
    s2 = KalmanSettings.from_mapping(OmegaConf.create({"n_states": 3, "n_obs": 2}))
    assert s2.n_states == 3
    with pytest.raises(ConfigurationError):
        KalmanSettings.from_mapping("not-a-mapping")  # type: ignore[arg-type]
    # from_hydra with overrides
    s3 = KalmanSettings.from_hydra(overrides=["n_states=4", "filter_type=ukf"])
    assert s3.n_states == 4 and s3.filter_type == "ukf"


@pytest.mark.unit
def test_trainer_filter_types_and_controls() -> None:
    trainer = KalmanTrainer(_settings(application="custom", filter_type="linear"))
    sys = trainer.build_system(n_states=2, n_obs=1)
    # with control matrix
    sys = LinearGaussianSSM(
        f=sys.f,
        h=sys.h,
        q=sys.q,
        r=sys.r,
        x0=sys.x0,
        p0=sys.p0,
        b=np.array([[0.1], [0.0]]),
        application="custom",
    )
    ctrl = np.ones((40, 1))
    states, obs = simulate_lds(sys, 40, rng=np.random.default_rng(1), controls=ctrl)
    for ft in ("linear", "ekf", "ukf", "adaptive"):
        tr = run_filter(obs, sys, _settings(filter_type=ft), controls=ctrl)  # type: ignore[arg-type]
        assert tr.means.shape[0] == 40
    result = trainer.fit(obs, system=sys, controls=ctrl)
    assert result.n_iter >= 1
    qh, rh = adapt_noise_from_trace(result.trace, sys)
    assert qh.shape == sys.q.shape


@pytest.mark.unit
def test_time_varying_matrices_and_ll_fallback() -> None:
    sys = build_system(_settings(application="denoise"), application="denoise")
    _, obs = simulate_lds(sys, 30, rng=np.random.default_rng(2))
    h_seq = np.stack([sys.h for _ in range(30)])
    f_seq = np.stack([sys.f for _ in range(30)])
    q_seq = np.stack([sys.q for _ in range(30)])
    r_seq = np.stack([sys.r for _ in range(30)])
    tr = filter_linear(obs, sys, h_seq=h_seq, f_seq=f_seq, q_seq=q_seq, r_seq=r_seq)
    assert np.isfinite(tr.log_likelihood)
    # singular S path
    ll = _gaussian_ll(np.array([1.0]), np.array([[0.0]]))
    assert np.isfinite(ll)
    # 1d observations
    tr2 = filter_linear(obs.reshape(-1), sys)
    assert tr2.means.shape[0] == 30


@pytest.mark.unit
def test_model_partial_fit_online_and_errors(tmp_path: Path) -> None:
    settings = _settings(application="denoise", online={"warm_start": True, "checkpoint_every": 2})
    model = KalmanFilterModel(settings=settings, random_seed=3)
    with pytest.raises(ValidationError):
        model.state()
    y = np.cumsum(np.random.default_rng(3).normal(0, 0.1, size=50))
    model.fit(y[:20])
    model.partial_fit(y[20:30])
    model.partial_fit(y[30:40])
    settings2 = _settings(
        application="denoise", online={"warm_start": False, "checkpoint_every": 0}
    )
    m2 = KalmanFilterModel(settings=settings2, random_seed=4)
    m2.fit(y[:10])
    m2.partial_fit(y[10:20])
    with pytest.raises(ValidationError):
        KalmanFilterModel(settings=settings).diagnostics()
    # dataframe extraction + close fallback
    frame = pl.DataFrame({"open_time": list(range(20)), "close": y[:20], "symbol": ["X"] * 20})
    m3 = KalmanFilterModel(settings=_settings(application="denoise"), random_seed=5)
    m3.fit(frame)
    assert m3.is_fitted
    # empty numeric cols without close
    with pytest.raises(ValidationError):
        KalmanFilterModel(settings=_settings())._extract_obs(pl.DataFrame({"symbol": ["a", "b"]}))
    # aic/bic
    assert np.isfinite(m3.aic(frame))
    assert np.isfinite(m3.bic(frame))
    # soft proba
    p = _soft_trend_proba(np.array([[1.0], [-1.0]]), np.stack([np.eye(1), np.eye(1)]))
    assert p.shape == (2, 2)
    # serializer json default
    assert isinstance(_json_default(np.array([1.0])), list)
    assert isinstance(_json_default(np.float64(1.0)), float)
    with pytest.raises(TypeError):
        _json_default(object())
    # visualization disabled
    off = _settings(visualization={"enabled": False, "max_points": 10})
    plot_filtered_state(np.zeros(5), tmp_path / "off.svg", off)
    assert (tmp_path / "off.svg").exists()
    _ensure(tmp_path / "x.svg", off)


@pytest.mark.unit
def test_dynamic_beta_and_spread() -> None:
    rng = np.random.default_rng(6)
    mkt = rng.normal(0, 1, size=80)
    beta = 1.2
    asset = 0.01 + beta * mkt + rng.normal(0, 0.05, size=80)
    y = np.column_stack([asset, mkt])
    settings = _settings(application="dynamic_beta", filter_type="linear")
    model = KalmanFilterModel(settings=settings, random_seed=6)
    model.fit(y)
    means = model.filtered_means()
    assert means.shape[1] == 2
    # beta roughly recovered
    assert abs(float(means[-1, 1]) - beta) < 0.5
    # spread / pairs
    for app in ("spread", "pairs"):
        s = _settings(application=app)
        m = KalmanFilterModel(settings=s, random_seed=7)
        m.fit(rng.normal(0, 1, size=40))
        assert m.is_fitted


@pytest.mark.unit
def test_diagnostics_evaluator_edge_cases() -> None:
    sys = build_system(_settings(application="denoise"), application="denoise")
    _, obs = simulate_lds(sys, 25, rng=np.random.default_rng(8))
    tr = filter_adaptive(obs, sys, window=5)
    sm = rts_smooth(tr, sys)
    diag = KalmanDiagnostics().report(sys, tr, smooth=sm, history=[-1.0, -0.5])
    assert diag["smoothed"]["final_mean"]
    assert _lag1_corr(np.array([1.0, 2.0])) == 0.0
    assert _lag1_corr(np.zeros(10)) == 0.0
    ev = KalmanEvaluator().evaluate(
        observations=obs, trace=tr, smooth=sm, true_states=np.zeros((25, 1)), n_params=3
    )
    assert "aic" in ev["metrics"]
    assert stats_chi2_crit(2) > 0
    # ukf singular-ish
    pts, wm, wc = sigma_points(np.zeros(2), 1e-18 * np.eye(2), alpha=1e-1)
    assert pts.shape[0] == 5
    # FilterTrace metadata path through serializer roundtrip
    model = KalmanFilterModel(settings=_settings(application="denoise"), system=sys, random_seed=9)
    model.fit(obs)
    model.smooth(obs)
    payload = model.export_state()
    m2 = KalmanFilterModel()
    m2.import_state(payload)
    assert m2.is_fitted
    # regime adapter state
    regime = KalmanRegimeModel(settings=_settings(application="denoise"), random_seed=10)
    frame = pl.DataFrame({"x": obs.reshape(-1)})
    regime.fit(frame, feature_columns=["x"])
    st = regime._algorithm_state()
    regime2 = KalmanRegimeModel(settings=_settings(application="denoise"))
    regime2._load_algorithm_state(st)
    assert regime2.is_fitted
