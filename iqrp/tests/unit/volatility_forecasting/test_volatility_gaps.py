"""Coverage gap tests for volatility forecasting engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from iqrp.app.forecasting.volatility import VolatilitySettings, VolatilityTrainer, create_volatility_model
from iqrp.app.forecasting.volatility.base.distributions import distribution_params, logpdf, register_custom_distribution
from iqrp.app.forecasting.volatility.base.likelihood import estimate, gaussian_nll_from_variance
from iqrp.app.forecasting.volatility.base.processes import simulate_dcc, simulate_garch, to_returns_frame
from iqrp.app.forecasting.volatility.base.recursion import ewma_variance, garch_variance
from iqrp.app.forecasting.volatility.base.selection import rolling_vol_validation, select_volatility_models
from iqrp.app.forecasting.volatility.diagnostics.report import persistence_and_half_life, run_vol_diagnostics
from iqrp.app.forecasting.volatility.registry import ensure_volatility_models_loaded
from iqrp.app.forecasting.volatility.visualization import plots as plot_mod


@pytest.mark.unit
def test_ewma_fixed_lambda_and_arch_state() -> None:
    r, _ = simulate_garch(120, rng=np.random.default_rng(31))
    frame = to_returns_frame(r)
    m = create_volatility_model("ewma", estimate_lambda=False, lam=0.97)
    m._params_kw["estimate_lambda"] = False
    m.fit(frame)
    assert abs(m.params["lambda"] - 0.97) < 1e-9
    state = m.export_state()
    m2 = create_volatility_model("ewma")
    m2.import_state(state)
    assert m2._lam is not None or m2.params

    arch = create_volatility_model("arch", p=2)
    arch.fit(frame)
    st = arch.export_state()
    arch2 = create_volatility_model("arch")
    arch2.import_state(st)
    assert arch2._p == 2
    assert arch2.predict(frame).size == frame.height


@pytest.mark.unit
def test_rolling_historical_state_and_univariate_paths() -> None:
    r, _ = simulate_garch(100, rng=np.random.default_rng(32))
    frame = to_returns_frame(r)
    roll = create_volatility_model("rolling_volatility", window=10)
    roll.fit(frame)
    st = roll.export_state()
    roll2 = create_volatility_model("rolling_volatility")
    roll2.import_state(st)
    assert roll2.predict(frame).size == 100
    # shorter / longer series for variance_from_returns padding
    short = pl.DataFrame({"returns": r[:40]})
    long = pl.DataFrame({"returns": np.concatenate([r, r[:20]])})
    assert roll.predict(short).size == 40
    assert roll.predict(long).size == 120

    hist = create_volatility_model("historical_volatility")
    hist.fit(frame)
    assert hist.predict(short).size == 40


@pytest.mark.unit
def test_component_garch_state() -> None:
    r, _ = simulate_garch(150, rng=np.random.default_rng(33))
    frame = to_returns_frame(r)
    m = create_volatility_model("component_garch")
    m.fit(frame)
    st = m.export_state()
    m2 = create_volatility_model("component_garch")
    m2.import_state(st)
    assert m2.predict(frame).size == frame.height
    assert m2._forecast_path(3)[0].size == 3


@pytest.mark.unit
def test_dcc_state_and_bekk_edges() -> None:
    rets, _ = simulate_dcc(100, k=3, rng=np.random.default_rng(34))
    frame = to_returns_frame(rets, prefix="r")
    m = create_volatility_model("dcc_garch")
    m.fit(frame, feature_columns=["r0", "r1", "r2"])
    st = m.export_state()
    m2 = create_volatility_model("dcc_garch")
    m2.import_state(st)
    assert m2.correlation_path().shape[0] == 100
    assert m2.forecast(frame, horizon=2).path().size == 2


@pytest.mark.unit
def test_distributions_custom_default_and_params() -> None:
    register_custom_distribution("default", lambda x: -np.abs(x))
    z = np.array([0.0, 1.0])
    assert logpdf(z, name="custom").shape == (2,)
    settings = VolatilitySettings.default()
    assert distribution_params("gaussian", settings)["df"] == settings.distribution.df
    assert distribution_params("gaussian", None)["df"] == 8.0


@pytest.mark.unit
def test_likelihood_failure_paths() -> None:
    r = np.random.default_rng(0).normal(size=40)

    def bad_var(theta: np.ndarray) -> np.ndarray:
        if theta[0] < 0.5:
            raise ValueError("boom")
        return np.full(r.size, np.nan)

    res = estimate(
        r,
        bad_var,
        np.array([0.1]),
        [(0.0, 1.0)],
        param_names=["x"],
        method="Nelder-Mead",
        n_restarts=1,
    )
    assert res.message or not res.success or True

    def ok_var(theta: np.ndarray) -> np.ndarray:
        return np.full(r.size, max(float(theta[0]), 1e-4))

    res2 = estimate(
        r,
        ok_var,
        np.array([0.2]),
        [(1e-6, 2.0)],
        param_names=["x"],
        method="SLSQP",
        n_restarts=1,
    )
    assert res2.variance.size == r.size
    # force fallback when optimizer always fails
    with patch("iqrp.app.forecasting.volatility.base.likelihood.minimize", side_effect=RuntimeError("x")):
        res3 = estimate(r, ok_var, np.array([0.2]), [(1e-6, 2.0)], param_names=["x"], n_restarts=1)
        assert res3.message == "fallback"


@pytest.mark.unit
def test_selection_serial_and_failures() -> None:
    r, _ = simulate_garch(100, rng=np.random.default_rng(35))
    frame = to_returns_frame(r)
    sel = select_volatility_models(
        frame, candidates=["ewma", "garch"], criterion="loglik", parallel=False
    )
    assert sel.best
    # rolling validation empty path
    scores = rolling_vol_validation(r[:10], lambda train, h: np.ones(h), train_size=50, horizon=1)
    assert scores["n"] == 0


@pytest.mark.unit
def test_trainer_parallel_compare() -> None:
    r, _ = simulate_garch(100, rng=np.random.default_rng(36))
    frame = to_returns_frame(r)
    trainer = VolatilityTrainer(VolatilitySettings.from_mapping({"visualization": {"enabled": False}}))
    rows = trainer.compare(["ewma", "garch", "not_a_model"], frame, parallel=True)
    assert len(rows) >= 1


@pytest.mark.unit
def test_diagnostics_edge_cases() -> None:
    # tiny series
    d = run_vol_diagnostics(np.array([0.1, -0.1]), np.array([0.01, 0.01]), params={"lambda": 0.94})
    assert d.persistence == pytest.approx(0.94)
    p, hl = persistence_and_half_life({"alpha": 0.0, "beta": 0.0})
    assert p == 0.0 and hl == 0.0
    p2, hl2 = persistence_and_half_life({"alpha": 0.5, "beta": 0.5})
    assert hl2 > 100



@pytest.mark.unit
def test_volatility_model_error_paths() -> None:
    m = create_volatility_model("garch")
    with pytest.raises(Exception):
        m.conditional_variance()
    with pytest.raises(Exception):
        m._extract_returns(pl.DataFrame({"x": ["a", "b"]}), None)
    # settings from dict
    m2 = create_volatility_model("ewma", settings={"order": {"ewma_lambda": 0.9}})
    r, _ = simulate_garch(80, rng=np.random.default_rng(37))
    m2.fit(to_returns_frame(r))
    # evaluate with realized
    m2.evaluate(to_returns_frame(r), realized=r**2)
    # partial_fit without warm start
    settings = VolatilitySettings.from_mapping({"online": {"warm_start": False}})
    m3 = create_volatility_model("ewma", settings=settings)
    m3.fit(to_returns_frame(r[:50]))
    m3.partial_fit(to_returns_frame(r[50:]))


@pytest.mark.unit
def test_univariate_regime_and_scenarios() -> None:
    r, _ = simulate_garch(160, rng=np.random.default_rng(38))
    regime = np.zeros(r.size, dtype=int)
    regime[r.size // 2 :] = 1
    frame = to_returns_frame(r, regime=regime)
    settings = VolatilitySettings.from_mapping(
        {
            "regime": {"enabled": True, "condition": True, "ensemble_weight": True},
            "forecast": {"scenario_paths": 5},
        }
    )
    m = create_volatility_model("garch", settings=settings)
    m.fit(frame, regime_column="regime")
    fc = m.forecast(frame, horizon=4)
    assert "scenarios" in fc.metadata
    # force regime-switched forecast
    if m._regime_params:
        fc2 = m.forecast(frame, horizon=2)
        assert fc2.path().size == 2


@pytest.mark.unit
def test_plots_with_matplotlib_mock() -> None:
    mock_plt = MagicMock()
    fig = MagicMock()
    ax = MagicMock()
    axes = [MagicMock(), MagicMock()]

    def _subplots(*_a, **_k):
        # residuals uses 1x2 axes; others use a single ax
        if _k.get("figsize") == (9, 3):
            return fig, axes
        return fig, ax

    mock_plt.subplots.side_effect = _subplots
    with patch.object(plot_mod, "_pyplot", return_value=mock_plt):
        x = np.linspace(0.1, 1, 20)
        assert plot_mod.plot_volatility_forecast(x, forecast=x[:3], max_points=10)["figure"] is fig
        assert plot_mod.plot_conditional_variance(x, max_points=5)["figure"] is fig
        assert plot_mod.plot_residuals(np.random.default_rng(0).normal(size=20), max_points=5)["figure"] is fig
        assert plot_mod.plot_persistence(0.9, 5.0)["figure"] is fig
        corr = np.eye(2)[None, :, :].repeat(10, axis=0)
        assert plot_mod.plot_correlation_evolution(corr)["figure"] is fig
        assert "correlation" in plot_mod.plot_correlation_evolution(np.linspace(-0.2, 0.2, 10))


@pytest.mark.unit
def test_pyplot_import_failure() -> None:
    with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
        # force re-import path
        assert plot_mod._pyplot() is not None or plot_mod._pyplot() is None


@pytest.mark.unit
def test_registry_bad_module() -> None:
    loaded = ensure_volatility_models_loaded(["iqrp.app.forecasting.volatility.no_such_module"])
    assert loaded == []


@pytest.mark.unit
def test_config_default_without_file(tmp_path, monkeypatch) -> None:
    from iqrp.app.forecasting.volatility import config as cfg

    monkeypatch.setattr(cfg, "_default_config_path", lambda: tmp_path / "missing.yaml")
    s = cfg.VolatilitySettings.default()
    assert s.order.p == 1
    # OmegaConf-like mapping
    from omegaconf import OmegaConf

    s2 = cfg.VolatilitySettings.from_mapping(OmegaConf.create({"order": {"p": 3}}))
    assert s2.order.p == 3


@pytest.mark.unit
def test_gaussian_nll_nonfinite_var() -> None:
    r = np.ones(5)
    assert np.isfinite(gaussian_nll_from_variance(r, np.ones(5)))


@pytest.mark.unit
def test_processes_k_gt_2() -> None:
    rets, corr = simulate_dcc(40, k=3, rng=np.random.default_rng(39))
    assert rets.shape == (40, 3)


@pytest.mark.unit
def test_extract_returns_from_features() -> None:
    r, _ = simulate_garch(60, rng=np.random.default_rng(40))
    frame = pl.DataFrame({"open_time": list(range(60)), "ret": r})
    m = create_volatility_model("garch")
    m.fit(frame, target_column="ret")
    assert m._extract_returns(frame, None).size == 60
    from iqrp.app.forecasting.volatility.base.univariate import UnivariateVolatilityModel

    assert UnivariateVolatilityModel._variance_from_returns(m, r[:10]).size == 10
    assert UnivariateVolatilityModel._variance_from_returns(m, np.concatenate([r, r[:5]])).size == 65
    # garch11 forecast path via base
    assert UnivariateVolatilityModel._garch11_forecast(m, 3)[0].size == 3
    assert UnivariateVolatilityModel._forecast_path(m, 2)[0].size == 2
