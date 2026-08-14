"""Second coverage pass for statistical forecasting (>98%)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.statistical import create_statistical_model
from iqrp.app.forecasting.statistical.ar.ar import ARModel
from iqrp.app.forecasting.statistical.arima.arima import ARIMAModel
from iqrp.app.forecasting.statistical.arma.arma import ARMAModel
from iqrp.app.forecasting.statistical.base.fitting import fit_arma_css, forecast_arma
from iqrp.app.forecasting.statistical.base.multivariate import johansen_trace
from iqrp.app.forecasting.statistical.base.processes import simulate_ar, simulate_var, to_frame
from iqrp.app.forecasting.statistical.base.selection import rolling_validation_score
from iqrp.app.forecasting.statistical.base.stationarity import (
    _lag_matrix,
    difference,
    integrate,
    suggest_differencing,
    suggest_seasonal_differencing,
)
from iqrp.app.forecasting.statistical.base.statistical_model import StatisticalForecastModel
from iqrp.app.forecasting.statistical.config import StatisticalSettings
from iqrp.app.forecasting.statistical.exponential.holt_winters import HoltWintersModel
from iqrp.app.forecasting.statistical.ma.ma import MAModel
from iqrp.app.forecasting.statistical.sarima.sarima import SARIMAModel
from iqrp.app.forecasting.statistical.trainer import StatisticalTrainResult
from iqrp.app.forecasting.statistical.var.var import VARModel
from iqrp.app.forecasting.statistical.varmax.varmax import VARMAXModel
from iqrp.app.forecasting.statistical.vecm.vecm import VECMModel
from iqrp.app.forecasting.statistical.visualization.charts import plot_forecast, plot_irf


@pytest.mark.unit
def test_save_load_all_heavy_models(tmp_path: Path) -> None:
    y = simulate_ar(100, [0.4], rng=np.random.default_rng(0))
    frame = to_frame(y).with_columns(pl.Series("x0", np.linspace(0, 1, 100)))
    settings = StatisticalSettings.from_mapping(
        {
            "identification": {"auto": True},
            "order": {
                "max_p": 1,
                "max_q": 1,
                "max_d": 0,
                "seasonal_period": 4,
                "max_P": 1,
                "max_D": 0,
                "max_Q": 0,
            },
            "columns": {"endogenous": ("target", "f0"), "exogenous": ("x0",), "target": "target"},
        }
    )
    cases = [
        ("ar", ARModel(settings=settings, p=None)),
        ("ma", MAModel(settings=settings, q=None)),
        ("arma", ARMAModel(settings=settings)),
        ("arima", ARIMAModel(settings=settings, p=1, d=0, q=0)),
        ("sarima", SARIMAModel(settings=settings, seasonal_period=4, p=1, d=0, q=0, P=1, D=0, Q=0)),
        ("holt_winters", HoltWintersModel(settings=settings, seasonal_period=4)),
        ("var", VARModel(settings=settings, p=None)),
        ("varmax", VARMAXModel(settings=settings, p=1, q=1)),
        ("vecm", VECMModel(settings=settings, lags=1)),
    ]
    for name, model in cases:
        if name in {"var", "varmax", "vecm"}:
            model.fit(frame, feature_columns=["target", "f0"], target_column="target")
        else:
            model.fit(frame, target_column="target")
        # force predict/forecast paths
        _ = model.predict(frame)
        _ = model.forecast(frame, horizon=2)
        if name == "sarima":
            # different length forces recompute predict
            _ = model.predict(frame.slice(0, 50))
        if name == "arima":
            # different length forces recompute
            model._fitted_values = None
            _ = model.predict(frame)
        if name == "holt_winters":
            model._fitted_values = None
            _ = model.predict(frame)
            st = model.export_state()
            m2 = HoltWintersModel(settings=settings)
            m2.import_state(st)
            assert m2.is_fitted
        if name == "var":
            st = model.export_state()
            m2 = VARModel(settings=settings)
            m2.import_state(st)
            assert m2.is_fitted
            _ = m2.predict(frame)
            _ = m2.forecast(frame, horizon=2)
        if name == "varmax":
            st = model.export_state()
            m2 = VARMAXModel(settings=settings)
            m2.import_state(st)
            assert m2.is_fitted
            _ = m2.predict(frame)
            _ = m2.forecast(frame, horizon=2)
        if name == "vecm":
            st = model.export_state()
            m2 = VECMModel(settings=settings)
            m2.import_state(st)
            assert m2.is_fitted
            _ = m2.predict(frame)
            _ = m2.forecast(frame, horizon=2)
        path = tmp_path / f"{name}.json"
        model.save(path)
        loaded = type(model).load(path)
        assert loaded.is_fitted


@pytest.mark.unit
def test_statistical_model_helpers() -> None:
    settings = StatisticalSettings.from_mapping({"identification": {"auto": False}})
    m = ARModel(settings=settings, p=1)
    # dict settings path
    m2 = ARModel(settings={"identification": {"auto": False}}, p=1)
    frame = to_frame(np.random.default_rng(0).normal(size=40))
    m.fit(frame, target_column="target")
    assert m.information_criteria is not None or m.order
    # evaluate ok
    m.evaluate(frame, target_column="target")
    # regime column missing path
    m._maybe_regime_series(frame, "nope")
    # extract target from features
    f2 = pl.DataFrame({"f0": [1.0, 2.0, 3.0]})
    assert m._extract_target(f2.with_columns(pl.Series("target", [1.0, 2.0, 3.0])), None).size == 3
    # frame from y with regime
    m._regime_column = "regime"
    templ = frame.with_columns(pl.Series("regime", [0, 1] * 20))
    rebuilt = m._frame_from_y(np.ones(10), templ, "target")
    assert "regime" in rebuilt.columns
    # short regime pad
    rebuilt2 = m._frame_from_y(np.ones(50), templ.slice(0, 5), "target")
    assert rebuilt2.height == 50
    # default horizon
    assert m._default_horizon(None) >= 1
    # TrainResult
    tr = StatisticalTrainResult("ar", {"p": 1}, {"rmse": 0.1})
    assert tr.to_dict()["model_name"] == "ar"


@pytest.mark.unit
def test_stationarity_lag_matrix_and_suggest() -> None:
    X, Y = _lag_matrix(np.arange(5.0), 0)
    assert Y.size == 0 or X.shape[0] == 0
    X, Y = _lag_matrix(np.arange(20.0), 3)
    assert Y.size > 0
    # suggest seasonal with strong seasonality
    t = np.arange(120)
    y = np.sin(2 * np.pi * t / 12) + 0.01 * np.random.default_rng(0).normal(size=120)
    suggest_seasonal_differencing(y, period=12, max_D=1)
    suggest_differencing(np.cumsum(np.random.default_rng(1).normal(size=100)), max_d=2)
    # integrate order 0
    assert integrate(np.array([1.0]), np.array([0.0]), order=0)[0] == 1.0


@pytest.mark.unit
def test_var_auto_endog_and_vecm_single() -> None:
    Y = simulate_var(80, np.array([[[0.3, 0.0], [0.0, 0.3]]]), rng=np.random.default_rng(2))
    frame = to_frame(Y, prefix="y")
    # no feature columns → auto numeric
    var = VARModel(
        settings=StatisticalSettings.from_mapping({"identification": {"auto": False}}), p=1
    )
    var.fit(frame, target_column="y0")
    assert var.predict(frame).shape[0] == frame.height
    # vecm with single column duplicates
    f1 = pl.DataFrame({"open_time": list(range(60)), "a": np.linspace(0, 1, 60)})
    vecm = VECMModel(lags=1)
    vecm.fit(f1, feature_columns=["a"])
    assert vecm.forecast(f1, horizon=2).values.size == 2
    # johansen failure path via singular
    johansen_trace(np.ones((30, 2)), lags=1)
    # arma nonfinite forecast path
    forecast_arma(np.array([1e10, 1e10]), np.zeros(2), np.array([2.0]), np.array([]), horizon=2)
    # fit_arma overflow clip path
    fit_arma_css(np.random.default_rng(0).normal(size=40) * 10, 2, 2)
    # rolling validation nan path already; force pad path
    rolling_validation_score(
        np.arange(30.0),
        lambda train, h: np.array([train[-1]]),
        train_size=10,
        horizon=3,
        step=10,
    )
    # viz lower/upper
    plot_forecast(
        np.arange(10.0),
        np.arange(10.0),
        Path("/tmp/fc_lu.svg"),
        lower=np.arange(10.0) - 1,
        upper=np.arange(10.0) + 1,
    )
    plot_irf(np.ones((5, 2, 2)), Path("/tmp/irf.svg"))


@pytest.mark.unit
def test_sarima_manual_order_and_omega() -> None:
    settings = StatisticalSettings.from_mapping(
        {"identification": {"auto": False}, "order": {"seasonal_period": 4}}
    )
    y = np.sin(np.linspace(0, 20, 80)) + np.random.default_rng(0).normal(0, 0.1, 80)
    frame = to_frame(y)
    m = SARIMAModel(settings=settings, p=1, d=0, q=0, P=0, D=0, Q=0, seasonal_period=4)
    m.fit(frame, target_column="target")
    m.predict(frame.slice(0, 40))
    st = m.export_state()
    m2 = SARIMAModel(settings=settings)
    m2.import_state(st)
    assert m2.forecast(frame, horizon=3).values.size == 3
    # varmax without exog names uses feature split
    settings2 = StatisticalSettings.from_mapping(
        {"identification": {"auto": False}, "order": {"p": 1}}
    )
    f2 = frame.with_columns(pl.Series("f1", np.linspace(0, 1, 80)))
    vx = VARMAXModel(settings=settings2, p=1, q=0)
    vx.fit(f2, feature_columns=["target", "f0", "f1"], target_column="target")
    assert vx.forecast(f2, horizon=2).values.size == 2
