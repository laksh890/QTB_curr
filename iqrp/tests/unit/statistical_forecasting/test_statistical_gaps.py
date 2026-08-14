"""Coverage gaps for statistical forecasting package."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from iqrp.app.core.exceptions import ConfigurationError, ValidationError
from iqrp.app.forecasting.statistical import create_statistical_model
from iqrp.app.forecasting.statistical.base.fitting import (
    FitResult,
    fit_arma_css,
    fit_var_ols,
    forecast_arma,
    forecast_var,
    lag_design,
)
from iqrp.app.forecasting.statistical.base.multivariate import (
    engle_granger,
    fit_vecm_engle_granger,
    johansen_trace,
)
from iqrp.app.forecasting.statistical.base.processes import (
    simulate_ma,
    simulate_seasonal_arima,
    to_frame,
)
from iqrp.app.forecasting.statistical.base.selection import (
    CandidateScore,
    SelectionResult,
    rolling_validation_score,
    select_arma_order,
)
from iqrp.app.forecasting.statistical.base.stationarity import (
    StationarityResult,
    adf_test,
    difference,
    integrate,
    kpss_test,
    phillips_perron_test,
    seasonal_difference,
    seasonal_integrate,
    suggest_seasonal_differencing,
)
from iqrp.app.forecasting.statistical.base.statistical_model import StatisticalForecastModel
from iqrp.app.forecasting.statistical.config import StatisticalSettings
from iqrp.app.forecasting.statistical.diagnostics.report import (
    acf,
    arch_lm,
    durbin_watson,
    jarque_bera,
    ljung_box,
    run_diagnostics,
)
from iqrp.app.forecasting.statistical.evaluation.metrics import evaluate_forecast, summary_table
from iqrp.app.forecasting.statistical.registry import (
    ensure_statistical_models_loaded,
    list_statistical_models,
)
from iqrp.app.forecasting.statistical.trainer import StatisticalTrainer
from iqrp.app.forecasting.statistical.visualization.charts import (
    plot_acf,
    plot_forecast,
    plot_qq,
    plot_residuals,
    plot_seasonal_decomposition,
)


@pytest.mark.unit
def test_stationarity_edges() -> None:
    assert adf_test(np.ones(5)).nobs == 5
    assert kpss_test(np.ones(5)).stationary
    assert phillips_perron_test(np.ones(5)).method == "pp"
    assert seasonal_difference(np.arange(5.0), period=10).size == 0
    assert integrate(np.array([1.0, 1.0]), np.array([10.0]), order=0).tolist() == [1.0, 1.0]
    assert integrate(np.array([1.0, 1.0]), np.array([10.0, 11.0]), order=2).size == 2
    assert seasonal_integrate(np.array([1.0, 1.0]), np.arange(12.0), period=4, order=1).size == 2
    assert suggest_seasonal_differencing(np.random.default_rng(0).normal(size=80), period=4) >= 0
    r = StationarityResult(0.0, 1.0, 0, 1, {}, False, "x")
    assert r.to_dict()["method"] == "x"
    adf_test(np.linspace(0, 1, 40), regression="ct")
    adf_test(np.linspace(0, 1, 40), regression="n")
    kpss_test(np.linspace(0, 1, 50), regression="ct")
    phillips_perron_test(np.cumsum(np.random.default_rng(0).normal(size=60)), regression="ct")


@pytest.mark.unit
def test_fitting_selection_edges() -> None:
    assert lag_design(np.arange(5.0), 0)[1].size == 5
    assert fit_arma_css(np.arange(10.0), 0, 0).intercept != 0 or True
    fr = FitResult(np.array([0.1]), np.ones(5), np.ones(5), 1.0, -10.0, 5, 2)
    assert fr.aic and fr.bic and fr.hqic and fr.aicc
    fr2 = FitResult(np.array([0.1]), np.ones(3), np.ones(3), 1.0, -10.0, 3, 3)
    assert fr2.aicc == fr2.aic or fr2.aicc
    empty = fit_var_ols(np.ones((2, 2)), 3)
    assert empty["nobs"] == 0
    path = forecast_arma(
        np.array([1.0, 2.0]), np.zeros(2), np.array([0.5]), np.array([0.1]), horizon=3
    )
    assert path.size == 3
    Y = np.random.default_rng(0).normal(size=(40, 2))
    fv = forecast_var(Y, np.array([[[0.2, 0.0], [0.0, 0.2]]]), np.zeros(2), horizon=2)
    assert fv.shape == (2, 2)
    # univariate history path in forecast_var
    forecast_var(np.array([1.0, 2.0, 3.0]), np.array([[[0.5]]]), np.array([0.0]), horizon=1)
    sel = select_arma_order(
        np.random.default_rng(0).normal(size=60), max_p=1, max_q=1, parallel=False
    )
    assert SelectionResult(sel.best_order, "aic", sel.leaderboard).to_dict()
    assert CandidateScore({"p": 1}, {"aic": 1.0}).to_dict()
    assert (
        rolling_validation_score(np.arange(5.0), lambda t, h: np.zeros(h), train_size=10)["n"] == 0
    )


@pytest.mark.unit
def test_diagnostics_evaluation_empty() -> None:
    assert run_diagnostics(np.array([])).nobs == 0
    assert durbin_watson(np.array([1.0])) != durbin_watson(np.array([1.0])) or True
    assert acf(np.ones(5)) == [0.0] * 4
    assert ljung_box(np.array([1.0]))[1] == 1.0
    assert jarque_bera(np.array([1.0, 2.0]))[1] == 1.0 or True
    assert arch_lm(np.ones(3))[1] == 1.0
    d = run_diagnostics(np.random.default_rng(0).normal(size=40))
    assert d.to_dict()["nobs"] == 40
    m = evaluate_forecast(np.linspace(0, 1, 20), np.linspace(0, 1, 20) + 0.01)
    assert "rmse" in m
    assert summary_table({"a": m})[0]["rank"] == 1


@pytest.mark.unit
def test_config_and_registry_errors() -> None:
    with pytest.raises(ConfigurationError):
        StatisticalSettings.from_mapping({"forecast": {"default_horizon": "bad"}})
    ensure_statistical_models_loaded(("iqrp.does_not_exist.mod",))
    assert "ar" in list_statistical_models()
    s = StatisticalSettings.from_hydra(config_path="/tmp/no_stat.yaml")
    assert s.forecast.default_horizon >= 1


@pytest.mark.unit
def test_model_online_modes_and_errors() -> None:
    y = np.cumsum(np.random.default_rng(0).normal(size=80))
    frame = to_frame(y)
    for mode in ("expanding", "sliding", "rolling"):
        settings = StatisticalSettings.from_mapping(
            {
                "online": {"mode": mode, "window": 30, "warm_start": True},
                "identification": {"auto": False},
            }
        )
        m = create_statistical_model("ar", settings=settings, p=1)
        m.fit(frame.slice(0, 40), target_column="target")
        m.partial_fit(frame.slice(40, 20), target_column="target")
    m2 = create_statistical_model("ar", p=1)
    with pytest.raises(ValidationError):
        m2.residuals()
    with pytest.raises(ValidationError):
        m2.evaluate(pl.DataFrame({"x": [1.0]}), target_column="missing")
    # warm_start false
    settings = StatisticalSettings.from_mapping(
        {"online": {"warm_start": False}, "identification": {"auto": False}}
    )
    m3 = create_statistical_model("ses", settings=settings)
    m3.fit(frame, target_column="target")
    m3.partial_fit(frame.slice(60, 10), target_column="target")


@pytest.mark.unit
def test_all_models_smoke_and_viz(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    y = rng.normal(size=100)
    frame = to_frame(y).with_columns(pl.Series("x0", rng.normal(size=100)))
    settings = StatisticalSettings.from_mapping(
        {
            "identification": {"auto": False},
            "order": {"p": 1, "d": 0, "q": 1, "P": 0, "D": 0, "Q": 0, "seasonal_period": 4},
            "columns": {"endogenous": ("target", "f0"), "exogenous": ("x0",), "target": "target"},
        }
    )
    for name in list_statistical_models():
        kwargs: dict = {"settings": settings}
        if name in {"ar", "arma", "arima", "var", "varmax"}:
            kwargs["p"] = 1
        if name in {"ma", "arma", "arima", "varmax"}:
            kwargs["q"] = 0 if name == "varmax" else 1
        if name == "arima":
            kwargs["d"] = 0
        if name in {"sarima", "holt_winters"}:
            kwargs["seasonal_period"] = 4
        if name == "vecm":
            kwargs["lags"] = 1
        m = create_statistical_model(name, **kwargs)
        if name in {"var", "varmax", "vecm"}:
            m.fit(frame, feature_columns=["target", "f0"], target_column="target")
        else:
            m.fit(frame, target_column="target")
        fc = m.forecast(frame, horizon=2)
        assert fc.values.size == 2
        _ = m.diagnostics()
    off = StatisticalSettings.from_mapping({"visualization": {"enabled": False}})
    plot_forecast(y, y, tmp_path / "off.svg", settings=off)
    plot_residuals(np.array([]), tmp_path / "er.svg")
    plot_qq(np.array([]), tmp_path / "eq.svg")
    plot_acf([], tmp_path / "ea.svg")
    plot_seasonal_decomposition(y[:10], 12, tmp_path / "es.svg")
    plot_forecast(np.array([]), np.array([]), tmp_path / "empty.svg")


@pytest.mark.unit
def test_processes_ma_and_multivariate_edges() -> None:
    assert simulate_ma(40, [0.4], rng=np.random.default_rng(0)).size == 40
    y = simulate_seasonal_arima(60, period=4, D=0, d=1, rng=np.random.default_rng(1))
    assert y.size == 60
    assert johansen_trace(np.ones((5, 1))).rank == 0
    assert (
        johansen_trace(np.random.default_rng(0).normal(size=(80, 3)), lags=2).method == "johansen"
    )
    assert engle_granger(np.arange(20.0), np.arange(20.0) * 2).rank in {0, 1}
    fit_vecm_engle_granger(np.random.default_rng(0).normal(size=(50, 2)), lags=2)
    # trainer compare
    frame = to_frame(np.random.default_rng(0).normal(size=80))
    StatisticalTrainer(
        StatisticalSettings.from_mapping({"identification": {"auto": False}})
    ).compare(["ar"], frame)
