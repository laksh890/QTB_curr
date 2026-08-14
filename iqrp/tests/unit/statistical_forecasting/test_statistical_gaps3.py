"""Final coverage push for statistical forecasting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from iqrp.app.core.exceptions import ValidationError
from iqrp.app.forecasting.statistical.ar.ar import ARModel
from iqrp.app.forecasting.statistical.arima.arima import ARIMAModel
from iqrp.app.forecasting.statistical.arma.arma import ARMAModel
from iqrp.app.forecasting.statistical.base.fitting import fit_ar_ols, fit_arma_css, fit_var_ols
from iqrp.app.forecasting.statistical.base.multivariate import (
    fit_vecm_engle_granger,
    granger_causality,
    impulse_response,
    johansen_trace,
)
from iqrp.app.forecasting.statistical.base.processes import to_frame
from iqrp.app.forecasting.statistical.base.stationarity import (
    adf_test,
    difference,
    kpss_test,
    phillips_perron_test,
    suggest_differencing,
    suggest_seasonal_differencing,
)
from iqrp.app.forecasting.statistical.config import StatisticalSettings
from iqrp.app.forecasting.statistical.ma.ma import MAModel
from iqrp.app.forecasting.statistical.var.var import VARModel
from iqrp.app.forecasting.statistical.vecm.vecm import VECMModel


@pytest.mark.unit
def test_arima_auto_and_predict_branches() -> None:
    y = np.cumsum(np.random.default_rng(0).normal(size=120))
    frame = to_frame(y)
    settings = StatisticalSettings.from_mapping(
        {"identification": {"auto": True}, "order": {"max_p": 1, "max_q": 0, "max_d": 1}}
    )
    m = ARIMAModel(settings=settings)
    m.fit(frame, target_column="target")
    m._fitted_values = None
    pred = m.predict(frame)
    assert pred.size == frame.height
    # d=0 path without fitted cache
    m0 = ARIMAModel(
        settings=StatisticalSettings.from_mapping({"identification": {"auto": False}}),
        p=1,
        d=0,
        q=0,
    )
    m0.fit(frame, target_column="target")
    m0._fitted_values = None
    assert m0.predict(frame).size == frame.height
    # arma/ma auto
    MAModel(settings=settings, q=None).fit(frame, target_column="target")
    ARMAModel(settings=settings).fit(frame, target_column="target")
    ARModel(settings=settings, p=None).fit(frame, target_column="target")


@pytest.mark.unit
def test_statistical_model_error_paths() -> None:
    m = ARModel(p=1)
    with pytest.raises(ValidationError):
        m._extract_target(pl.DataFrame({"x": ["a", "b"]}), None)
    with pytest.raises(ValidationError):
        m._resolve_target_name(pl.DataFrame({"x": ["a"]}), None)
    # partial_fit when not fitted
    frame = to_frame(np.random.default_rng(0).normal(size=40))
    m.partial_fit(frame, target_column="target")
    assert m.is_fitted
    # diagnostics empty residuals
    m._residuals = None
    assert m.residuals().size == 0
    # evaluate missing target
    with pytest.raises(ValidationError):
        m.evaluate(pl.DataFrame({"z": [1.0]}), target_column="nope")
    # regime conditioning
    y = np.arange(20.0)
    reg = np.array([0] * 10 + [1] * 10)
    out = m._regime_conditioned_y(y, reg)
    assert out.size == 20
    # omega conf mapping
    cfg = OmegaConf.create({"forecast": {"default_horizon": 3}})
    assert StatisticalSettings.from_mapping(cfg).forecast.default_horizon == 3
    # default() missing file
    with patch("iqrp.app.forecasting.statistical.config._default_config_path") as p:
        p.return_value = __import__("pathlib").Path("/tmp/missing_stat.yaml")
        assert StatisticalSettings.default().forecast.default_horizon >= 1


@pytest.mark.unit
def test_fitting_and_multivariate_edges() -> None:
    assert fit_ar_ols(np.array([]), 2).nobs == 0
    # arma css import fail path already; force minimize fail
    with patch("iqrp.app.forecasting.statistical.base.fitting.minimize") as mim:
        mim.return_value = MagicMock(success=False, x=np.array([0.0, 0.1, 0.0]))
        fit_arma_css(np.random.default_rng(0).normal(size=30), 1, 1)
    fit = fit_var_ols(np.random.default_rng(0).normal(size=(40, 2)), 1)
    # bad sigma slogdet
    bad = fit.copy()
    # impulse response cholesky fail
    impulse_response(
        np.array([[[0.1, 0.0], [0.0, 0.1]]]), np.array([[1.0, 2.0], [2.0, 1.0]]), horizon=3
    )
    # granger edges
    assert granger_causality(np.ones((5, 1)), cause=0, effect=0).pvalue == 1.0
    assert (
        johansen_trace(np.random.default_rng(0).normal(size=(10, 2)), lags=5).method == "johansen"
    )
    # vecm K=1 path inside fit_vecm
    fit_vecm_engle_granger(np.random.default_rng(0).normal(size=(40, 1)), lags=1)
    # stationarity short series branches already; hit suggest seasonal small
    suggest_seasonal_differencing(np.arange(10.0), period=12, max_D=1)
    suggest_differencing(np.arange(10.0), max_d=2)
    adf_test(np.arange(12.0), max_lags=5)
    # kpss ones
    kpss_test(np.linspace(0, 1, 30))
    phillips_perron_test(np.linspace(0, 1, 30))


@pytest.mark.unit
def test_var_endog_settings_and_vecm_forecast_pad() -> None:
    settings = StatisticalSettings.from_mapping(
        {
            "identification": {"auto": True},
            "order": {"max_var_lags": 2},
            "columns": {"endogenous": ("y0", "y1"), "target": "y0"},
        }
    )
    Y = np.random.default_rng(0).normal(size=(60, 2))
    frame = pl.DataFrame({"open_time": list(range(60)), "y0": Y[:, 0], "y1": Y[:, 1]})
    var = VARModel(settings=settings, p=None)
    var.fit(frame)
    # pad residuals path with larger p relative — already
    # vecm forecast with lag padding
    vecm = VECMModel(lags=3)
    vecm.fit(frame, feature_columns=["y0", "y1"])
    # force B mismatch pad in forecast
    fc = vecm.forecast(frame, horizon=3)
    assert fc.values.size == 3
    # fitted values size match predict
    assert vecm.predict(frame).size == frame.height
    # predict mismatch size
    vecm._fitted_values = np.ones(3)
    assert vecm.predict(frame).size == frame.height
