"""Multivariate volatility model tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from iqrp.app.forecasting.volatility import create_volatility_model
from iqrp.app.forecasting.volatility.base.processes import simulate_dcc, to_returns_frame


@pytest.mark.unit
def test_dcc_garch_api() -> None:
    rets, _ = simulate_dcc(160, k=2, rng=np.random.default_rng(11))
    frame = to_returns_frame(rets, prefix="r")
    model = create_volatility_model("dcc_garch")
    model.fit(frame, feature_columns=["r0", "r1"])
    assert model.conditional_volatility().size == frame.height
    assert model.predict(frame).size == frame.height
    cov = model.forecast_covariance(horizon=3)
    assert cov.shape == (3, 2, 2)
    fc = model.forecast(frame, horizon=2)
    assert "covariance" in fc.metadata
    corr = model.correlation_path()
    assert corr.shape[0] == frame.height
    # single-column fallback
    uni = pl.DataFrame({"returns": rets[:, 0]})
    m2 = create_volatility_model("dcc_garch")
    m2.fit(uni)
    assert m2.is_fitted


@pytest.mark.unit
def test_bekk_api() -> None:
    rets, _ = simulate_dcc(140, k=2, rng=np.random.default_rng(12))
    frame = to_returns_frame(rets, prefix="a")
    model = create_volatility_model("bekk")
    model.fit(frame, feature_columns=["a0", "a1"])
    cov = model.forecast_covariance(horizon=4)
    assert cov.shape[0] == 4
    assert model.predict(frame).size == frame.height
    state = model.export_state()
    m2 = create_volatility_model("bekk")
    m2.import_state(state)
    assert m2.is_fitted
    # univariate fallback
    m3 = create_volatility_model("bekk")
    m3.fit(pl.DataFrame({"returns": rets[:, 0]}))
    assert m3.forecast(pl.DataFrame({"returns": rets[:, 0]}), horizon=2).horizon == 2
