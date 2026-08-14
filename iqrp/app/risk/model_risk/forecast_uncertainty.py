"""Forecast uncertainty monitor."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_returns


def forecast_uncertainty(
    forecasts: Any,
    realizations: Any,
    *,
    window: int | None = None,
) -> RiskMeasure:
    """RMSE of forecast errors on aligned past pairs only (no look-ahead)."""
    f = as_returns(forecasts)
    y = as_returns(realizations)
    n = int(min(f.size, y.size))
    if n == 0:
        return RiskMeasure(
            name="forecast_uncertainty",
            value=0.0,
            unit="rmse",
            method="forecast_rmse",
            parameters={"n_obs": 0, "window": window},
        )
    f = f[-n:]
    y = y[-n:]
    if window is not None and window > 0:
        w = min(int(window), n)
        f = f[-w:]
        y = y[-w:]
        n = w
    err = y - f
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    return RiskMeasure(
        name="forecast_uncertainty",
        value=rmse,
        unit="rmse",
        method="forecast_rmse",
        parameters={"n_obs": n, "window": window, "mae": mae, "bias": bias},
    )
