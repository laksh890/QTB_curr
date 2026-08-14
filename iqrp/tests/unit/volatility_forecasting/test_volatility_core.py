"""Core unit tests for Institutional Volatility Forecasting Engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.forecasting.volatility import (
    VolatilitySettings,
    VolatilityTrainer,
    create_volatility_model,
    list_volatility_models,
)
from iqrp.app.forecasting.volatility.base.distributions import (
    logpdf,
    register_custom_distribution,
)
from iqrp.app.forecasting.volatility.base.likelihood import estimate, gaussian_nll_from_variance
from iqrp.app.forecasting.volatility.base.processes import (
    simulate_dcc,
    simulate_garch,
    simulate_gjr,
    to_returns_frame,
)
from iqrp.app.forecasting.volatility.base.recursion import (
    aparch_variance,
    arch_variance,
    cgarch_variance,
    egarch_variance,
    ewma_variance,
    figarch_variance,
    forecast_garch_path,
    garch_variance,
    gjr_variance,
)
from iqrp.app.forecasting.volatility.base.selection import (
    rolling_vol_validation,
    select_volatility_models,
)
from iqrp.app.forecasting.volatility.diagnostics import run_vol_diagnostics
from iqrp.app.forecasting.volatility.evaluation.metrics import (
    evaluate_volatility,
    mae,
    mse,
    qlike,
    realized_volatility,
    rmse,
)
from iqrp.app.forecasting.volatility.visualization import (
    plot_conditional_variance,
    plot_correlation_evolution,
    plot_persistence,
    plot_residuals,
    plot_volatility_forecast,
)


@pytest.fixture
def garch_frame() -> pl.DataFrame:
    r, _ = simulate_garch(280, omega=0.05, alpha=0.1, beta=0.85, rng=np.random.default_rng(7))
    regimes = (np.arange(r.size) > r.size // 2).astype(int)
    return to_returns_frame(r, regime=regimes)


@pytest.mark.unit
def test_registry_lists_all_models() -> None:
    names = set(list_volatility_models())
    assert names >= {
        "historical_volatility",
        "rolling_volatility",
        "ewma",
        "arch",
        "garch",
        "egarch",
        "gjr_garch",
        "figarch",
        "aparch",
        "component_garch",
        "dcc_garch",
        "bekk",
    }


@pytest.mark.unit
def test_settings_hydra_and_mapping() -> None:
    s = VolatilitySettings.default()
    assert s.order.p == 1
    s2 = VolatilitySettings.from_mapping({"order": {"p": 2}, "distribution": {"name": "student_t"}})
    assert s2.order.p == 2
    s3 = VolatilitySettings.from_hydra(overrides=["forecast.default_horizon=7"])
    assert s3.forecast.default_horizon == 7
    with pytest.raises(Exception):
        VolatilitySettings.from_mapping({"distribution": {"name": "not_a_dist"}})


@pytest.mark.unit
def test_distributions_and_custom() -> None:
    z = np.linspace(-2, 2, 50)
    for name in ("gaussian", "student_t", "skew_t", "ged", "laplace"):
        assert np.all(np.isfinite(logpdf(z, name=name)))
    register_custom_distribution("unit_test_dist", lambda x: -0.5 * x**2)
    assert logpdf(z, name="unit_test_dist").shape == z.shape
    assert logpdf(z, name="unknown_fallback").shape == z.shape


@pytest.mark.unit
def test_recursions_and_forecast_path() -> None:
    eps = np.random.default_rng(0).normal(size=100)
    assert ewma_variance(eps, 0.94).size == 100
    assert arch_variance(eps, 0.1, np.array([0.2])).min() > 0
    assert garch_variance(eps, 0.1, np.array([0.05]), np.array([0.9])).min() > 0
    assert gjr_variance(eps, 0.1, np.array([0.05]), np.array([0.1]), np.array([0.85])).min() > 0
    assert (
        egarch_variance(eps, -0.1, np.array([0.1]), np.array([-0.05]), np.array([0.95])).min() > 0
    )
    assert (
        aparch_variance(eps, 0.1, np.array([0.1]), np.array([0.1]), np.array([0.8]), 2.0).min() > 0
    )
    h, q = cgarch_variance(eps, 0.05, 0.95, 0.05, 0.05, 0.8)
    assert h.size == q.size
    assert figarch_variance(eps, 0.05, 0.2, 0.4, 0.4).min() > 0
    path = forecast_garch_path(0.1, 0.2, 0.05, 0.1, 0.85, horizon=5)
    assert path.shape == (5,)


@pytest.mark.unit
def test_likelihood_estimate() -> None:
    r, _ = simulate_garch(150, rng=np.random.default_rng(1))

    def var_fn(theta: np.ndarray) -> np.ndarray:
        return garch_variance(r, float(theta[0]), np.array([theta[1]]), np.array([theta[2]]))

    res = estimate(
        r,
        var_fn,
        np.array([0.05, 0.1, 0.85]),
        [(1e-8, 1.0), (0.0, 1.0), (0.0, 1.0)],
        param_names=["omega", "alpha", "beta"],
        method="robust",
        n_restarts=2,
    )
    assert res.variance.size == r.size
    assert "aic" in res.to_dict()
    nll = gaussian_nll_from_variance(r, res.variance, dist="student_t", dist_kwargs={"df": 8.0})
    assert np.isfinite(nll)


@pytest.mark.unit
def test_metrics_and_diagnostics() -> None:
    r, v = simulate_garch(120, rng=np.random.default_rng(2))
    m = evaluate_volatility(r, v)
    assert set(m) >= {"qlike", "rmse", "mae", "mse", "loglik"}
    assert qlike(r**2, v) > 0
    assert rmse(r**2, v) >= 0 and mae(r**2, v) >= 0 and mse(r**2, v) >= 0
    assert realized_volatility(r).size == r.size
    diag = run_vol_diagnostics(r, v, params={"alpha": 0.1, "beta": 0.85})
    assert diag.persistence > 0
    assert "half_life" in diag.to_dict()


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        "historical_volatility",
        "rolling_volatility",
        "ewma",
        "arch",
        "garch",
        "egarch",
        "gjr_garch",
        "figarch",
        "aparch",
        "component_garch",
    ],
)
def test_univariate_models_api(name: str, garch_frame: pl.DataFrame) -> None:
    model = create_volatility_model(name)
    model.fit(garch_frame, target_column="returns", regime_column="regime")
    assert model.is_fitted
    sigma = model.predict(garch_frame)
    assert sigma.shape[0] == garch_frame.height
    assert model.conditional_variance().min() > 0
    assert model.annualized_volatility().mean() > 0
    fc = model.forecast(garch_frame, horizon=4)
    assert fc.path().shape == (4,)
    assert "variance" in fc.metadata
    intervals = model.forecast_interval(garch_frame, horizon=3)
    assert len(intervals) == 3
    cov = model.forecast_covariance(horizon=2)
    assert cov.size >= 2
    report = model.evaluate(garch_frame)
    assert report.n_samples > 0
    d = model.diagnostics()
    assert d.mean_variance > 0
    # OOS predict
    r2 = garch_frame["returns"].to_numpy() * 1.01
    other = pl.DataFrame({"open_time": list(range(len(r2))), "returns": r2})
    assert model.predict(other).size == len(r2)


@pytest.mark.unit
def test_partial_fit_and_serialization(garch_frame: pl.DataFrame, tmp_path: Path) -> None:
    model = create_volatility_model("ewma")
    mid = garch_frame.height // 2
    model.fit(garch_frame[:mid])
    model.partial_fit(garch_frame[mid:])
    assert model.is_fitted
    path = tmp_path / "ewma.joblib"
    model.save(path)
    loaded = type(model).load(path)
    assert loaded.is_fitted
    assert np.allclose(loaded.conditional_volatility(), model.conditional_volatility())
    ck = model.checkpoint()
    model2 = create_volatility_model("ewma")
    model2.import_state(ck)
    assert model2.is_fitted


@pytest.mark.unit
def test_selection_and_rolling_validation(garch_frame: pl.DataFrame) -> None:
    sel = select_volatility_models(
        garch_frame,
        candidates=["ewma", "arch", "garch"],
        criterion="aic",
        parallel=True,
    )
    assert sel.best in {"ewma", "arch", "garch"}
    assert sel.to_dict()["leaderboard"]
    r = garch_frame["returns"].to_numpy()

    def fn(train: np.ndarray, h: int) -> np.ndarray:
        v = ewma_variance(train, 0.94)
        return np.full(h, v[-1])

    scores = rolling_vol_validation(r, fn, train_size=80, horizon=1, step=20)
    assert scores["n"] > 0


@pytest.mark.unit
def test_trainer_compare_and_auto(garch_frame: pl.DataFrame) -> None:
    settings = VolatilitySettings.from_mapping(
        {"visualization": {"enabled": True}, "forecast": {"scenario_paths": 3}}
    )
    trainer = VolatilityTrainer(settings)
    model, result = trainer.fit("garch", garch_frame)
    assert result.params
    assert result.to_dict()["diagnostics"]
    rows = trainer.compare(["ewma", "garch"], garch_frame, parallel=False)
    assert len(rows) >= 1
    m2, res2 = trainer.auto_select(garch_frame, candidates=["ewma", "garch"])
    assert res2.selection is not None
    assert m2.meta.name in {"ewma", "garch"}


@pytest.mark.unit
def test_visualization_helpers() -> None:
    x = np.linspace(0.1, 1.0, 40)
    assert "in_sample" in plot_volatility_forecast(x, forecast=x[:5])
    assert "variance" in plot_conditional_variance(x**2)
    assert "residuals" in plot_residuals(np.random.default_rng(0).normal(size=40))
    assert "half_life" in plot_persistence(0.9, 6.5)
    corr = np.random.default_rng(0).uniform(-0.5, 0.5, size=(30, 2, 2))
    for i in range(30):
        corr[i] = np.eye(2)
        corr[i, 0, 1] = corr[i, 1, 0] = 0.3
    assert "correlation" in plot_correlation_evolution(corr)


@pytest.mark.unit
def test_processes_gjr_dcc() -> None:
    r, v = simulate_gjr(100, rng=np.random.default_rng(3))
    assert r.size == v.size == 100
    rets, corr = simulate_dcc(80, k=2, rng=np.random.default_rng(4))
    assert rets.shape == (80, 2)
    assert corr.size == 80
    frame = to_returns_frame(rets, prefix="r")
    assert "r0" in frame.columns
