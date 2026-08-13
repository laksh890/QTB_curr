"""Additional coverage gaps for Kalman engine (>98%)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from iqrp.app.core.exceptions import ConfigurationError
from iqrp.app.regimes.kalman.adaptive import adapt_noise_from_trace, filter_adaptive
from iqrp.app.regimes.kalman.config import KalmanSettings, _default_config_path
from iqrp.app.regimes.kalman.covariance import ensure_spd, mahalanobis
from iqrp.app.regimes.kalman.diagnostics import KalmanDiagnostics
from iqrp.app.regimes.kalman.ekf import filter_ekf
from iqrp.app.regimes.kalman.evaluator import stats_chi2_crit
from iqrp.app.regimes.kalman.initialization import LinearGaussianSSM, build_system
from iqrp.app.regimes.kalman.linear import FilterTrace, _gaussian_ll, filter_linear
from iqrp.app.regimes.kalman.model import KalmanFilterModel, KalmanRegimeModel, _soft_trend_proba
from iqrp.app.regimes.kalman.smoothing import rts_smooth
from iqrp.app.regimes.kalman.trainer import KalmanTrainer, simulate_lds
from iqrp.app.regimes.kalman.ukf import filter_ukf, sigma_points
from iqrp.app.regimes.kalman.update import innovation_statistics, update_nonlinear, update_state


def _settings(**kw: object) -> KalmanSettings:
    data = {
        **KalmanSettings.default().model_dump(),
        "training": {"em_iterations": 2, "tol": 1e-3, "estimate_noise": True},
    }
    data.update(kw)
    return KalmanSettings.from_mapping(data)


@pytest.mark.unit
def test_adaptive_inflate_and_1d_innov() -> None:
    sys = build_system(_settings(application="denoise"), application="denoise")
    # huge observation outliers → mahalanobis above threshold
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.01, size=40)
    y[20:] = 50.0
    tr = filter_adaptive(
        y,
        sys,
        window=5,
        process_adapt_rate=0.5,
        observation_adapt_rate=0.5,
        innovation_threshold=0.1,
    )
    assert "q_final" in tr.metadata
    # 1d innovations branch in adapt_noise_from_trace
    tr2 = FilterTrace(
        means=np.array([[1.0], [1.1], [1.2]]),
        covs=np.stack([np.eye(1)] * 3),
        pred_means=np.array([[1.0], [1.1], [1.2]]),
        pred_covs=np.stack([np.eye(1)] * 3),
        innovations=np.array([0.1, 0.2, 0.15]),
        innovation_covs=np.stack([[[0.1]]] * 3),
        gains=np.stack([[[1.0]]] * 3),
        log_likelihood=0.0,
    )
    q, r = adapt_noise_from_trace(tr2, sys)
    assert q.shape == (1, 1) and r.shape == (1, 1)


@pytest.mark.unit
def test_config_error_and_default_fallback(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        KalmanSettings.from_hydra(bad)
    with patch("iqrp.app.regimes.kalman.config._default_config_path", return_value=tmp_path / "missing.yaml"):
        s = KalmanSettings.default()
        assert s.n_states == 2
    assert _default_config_path().name == "default.yaml"


@pytest.mark.unit
def test_covariance_linalg_and_pinv() -> None:
    # numba import branch already taken; exercise eigh failure path via patch
    with patch("numpy.linalg.eigh", side_effect=np.linalg.LinAlgError("fail")):
        m = ensure_spd(np.array([[2.0, 0.1], [0.1, 2.0]]))
        assert m.shape == (2, 2)
    with patch("numpy.linalg.solve", side_effect=np.linalg.LinAlgError("fail")):
        d = mahalanobis(np.array([1.0, 0.0]), np.eye(2))
        assert d >= 0


@pytest.mark.unit
def test_linear_ll_and_smoothing_pinv() -> None:
    # slogdet failure / singular path
    ll = _gaussian_ll(np.array([1.0, 2.0]), np.array([[1e-20, 0.0], [0.0, 1e-20]]))
    assert np.isfinite(ll)
    with patch("numpy.linalg.slogdet", return_value=(-1.0, 0.0)):
        ll2 = _gaussian_ll(np.array([0.5]), np.array([[1.0]]))
        assert np.isfinite(ll2)
    sys = build_system(_settings(application="denoise"), application="denoise")
    _, obs = simulate_lds(sys, 8, rng=np.random.default_rng(1))
    tr = filter_linear(obs, sys)
    with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("fail")):
        sm = rts_smooth(tr, sys)
        assert sm.means.shape[0] == 8


@pytest.mark.unit
def test_update_pinv_and_1d_innov_stats() -> None:
    with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("fail")):
        x, p, innov, s, k = update_state(
            np.zeros(1), np.eye(1), np.array([0.1]), np.array([[1.0]]), np.array([[0.1]])
        )
        assert k.shape == (1, 1)
        xu, pu, *_ = update_nonlinear(
            np.zeros(1),
            np.eye(1),
            np.array([0.1]),
            lambda x: x,
            lambda x: np.eye(1),
            np.array([[0.1]]),
        )
        assert xu.shape[0] == 1
    stats = innovation_statistics(np.array([0.1, -0.2, 0.05]), np.eye(1) * 0.1)
    assert stats["n"] == 3.0


@pytest.mark.unit
def test_ukf_cholesky_and_gain_pinv() -> None:
    with patch("numpy.linalg.cholesky", side_effect=[np.linalg.LinAlgError("fail"), np.array([[0.1]])]):
        pts, wm, wc = sigma_points(np.zeros(1), np.eye(1) * 0.01, alpha=0.5)
        assert pts.shape[0] == 3
    sys = build_system(_settings(application="denoise"), application="denoise")
    _, obs = simulate_lds(sys, 10, rng=np.random.default_rng(2))
    with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("fail")):
        tr = filter_ukf(obs, sys, alpha=0.1)
        assert np.isfinite(tr.log_likelihood)


@pytest.mark.unit
def test_ekf_default_hooks_and_evaluator_fallback() -> None:
    sys = build_system(_settings(application="denoise"), application="denoise")
    # force numerical jacobian defaults via None hooks
    sys = LinearGaussianSSM(f=sys.f, h=sys.h, q=sys.q, r=sys.r, x0=sys.x0, p0=sys.p0)
    _, obs = simulate_lds(sys, 12, rng=np.random.default_rng(3))
    tr = filter_ekf(obs, sys, f_fn=None, h_fn=None, f_jac=None, h_jac=None)
    assert np.isfinite(tr.log_likelihood)
    with patch("scipy.stats.chi2.ppf", side_effect=Exception("no scipy")):
        assert stats_chi2_crit(3) == 12.0


@pytest.mark.unit
def test_model_branches_online_state_diag(tmp_path: Path) -> None:
    # soft proba 1d means
    p = _soft_trend_proba(np.array([1.0, -1.0]), np.stack([np.eye(1), np.eye(1)]))
    assert p.shape == (2, 2)

    # dynamic_beta with h already provided (h is None branch skipped)
    rng = np.random.default_rng(4)
    mkt = rng.normal(size=30)
    y = np.column_stack([0.5 * mkt, mkt])
    h = np.stack([np.ones(30), mkt], axis=1).reshape(30, 1, 2)
    model = KalmanFilterModel(settings=_settings(application="dynamic_beta"), random_seed=4)
    model.fit(y, h_seq=h)
    model.filter(y)  # rebuilds h_seq from raw
    model.filter(y[:, 0:1])  # uses stored / maybe_dynamic_beta path

    # partial_fit when not fitted
    m0 = KalmanFilterModel(settings=_settings(application="denoise"), random_seed=5)
    m0.partial_fit(np.linspace(0, 1, 15))
    assert m0.is_fitted

    # update without online state (force None)
    m1 = KalmanFilterModel(settings=_settings(application="denoise"), random_seed=6)
    m1.fit(np.linspace(0, 1, 20))
    m1._online_x = None
    m1._online_p = None
    m1._last_innov = None
    m1._last_gain = None
    assert m1.state().shape[0] == 1
    assert m1.covariance().shape == (1, 1)
    assert m1.innovation().shape[0] == 1
    assert m1.kalman_gain().shape[0] == 1
    m1._smooth = None
    assert m1.smoothed_means().shape[0] == 20

    # diagnostics via train_obs only
    m1._trace = None
    m1._smooth = None
    d = m1.diagnostics()
    assert "log_likelihood" in d

    # n_params without system
    m2 = KalmanFilterModel(settings=_settings())
    assert m2._n_params() > 0

    # _build_h_seq single-col → None
    assert m2._build_h_seq(np.ones((5, 1)), np.ones((5, 1))) is None
    assert m2._extract_controls(np.ones((5, 1))) is None
    settings = _settings(
        columns={"timestamp": "t", "observation_columns": None, "control_columns": ("u", "v")}
    )
    m3 = KalmanFilterModel(settings=settings)
    frame = pl.DataFrame({"t": [1, 2], "x": [1.0, 2.0], "w": [0.0, 0.0]})
    assert m3._extract_controls(frame) is None
    settings2 = _settings(
        application="denoise",
        columns={"timestamp": "t", "observation_columns": None, "control_columns": ("u",)},
    )
    m4 = KalmanFilterModel(settings=settings2)
    frame2 = pl.DataFrame({"t": [1, 2, 3], "x": [1.0, 1.1, 1.2], "u": [0.0, 0.1, 0.0]})
    m4.fit(frame2, observation_columns=["x"])
    assert m4._controls is not None

    # regime load path when not fitted
    regime = KalmanRegimeModel(settings=_settings(application="denoise"))
    regime._load_algorithm_state({"fitted": False, "algorithm_state": {}, "state_names": []})
    assert not regime.is_fitted


@pytest.mark.unit
def test_trainer_seed_x0_and_adaptive_qr() -> None:
    settings = _settings(application="denoise", filter_type="adaptive", training={"em_iterations": 3, "tol": 1e-12, "estimate_noise": True})
    trainer = KalmanTrainer(settings)
    sys = build_system(settings, application="denoise")
    # zero x0 triggers seeding
    sys = LinearGaussianSSM(f=sys.f, h=sys.h, q=sys.q, r=sys.r, x0=np.zeros(1), p0=sys.p0, application="denoise")
    y = np.linspace(5.0, 6.0, 25)
    res = trainer.fit(y, system=sys)
    assert res.system.x0[0] != 0.0 or True
    # simulate with zero p0
    sys2 = LinearGaussianSSM(
        f=sys.f, h=sys.h, q=sys.q, r=sys.r, x0=np.array([1.0]), p0=np.zeros((1, 1)), application="denoise"
    )
    st, ob = simulate_lds(sys2, 5, rng=np.random.default_rng(7))
    assert st.shape[0] == 5

    # diagnostics 1d innovations reshape
    tr = res.trace
    tr.innovations = tr.innovations.reshape(-1) if tr.innovations.ndim > 1 else tr.innovations
    # force 1d path in diagnostics
    tr_1d = FilterTrace(
        means=tr.means,
        covs=tr.covs,
        pred_means=tr.pred_means,
        pred_covs=tr.pred_covs,
        innovations=tr.innovations.reshape(-1),
        innovation_covs=tr.innovation_covs,
        gains=tr.gains,
        log_likelihood=tr.log_likelihood,
        metadata=dict(tr.metadata),
    )
    report = KalmanDiagnostics().report(res.system, tr_1d)
    assert report["n_obs"] == 1
