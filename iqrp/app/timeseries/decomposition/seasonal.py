"""Seasonal component extraction helpers."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array
from iqrp.app.timeseries.decomposition.classical import classical_decompose


def extract_seasonal(
    x: np.ndarray | list[float],
    *,
    period: int = 24,
    model: str = "additive",
) -> AnalysisResult:
    res = classical_decompose(x, period=period, model=model)
    strength = _seasonal_strength(res.seasonal, res.residual)
    return AnalysisResult(
        method="seasonal.extract",
        value=res.seasonal,
        parameters={"period": period, "model": model},
        temporal_mode=TemporalMode.FULL_SAMPLE,
        metadata={"seasonal_strength": strength},
    )


def seasonal_strength(seasonal: np.ndarray, residual: np.ndarray) -> float:
    return _seasonal_strength(seasonal, residual)


def _seasonal_strength(seasonal: np.ndarray, residual: np.ndarray) -> float:
    s = as_float_array(seasonal)
    r = as_float_array(residual)
    var_r = float(np.nanvar(r))
    var_sr = float(np.nanvar(s + r))
    if var_sr <= 1e-15:
        return 0.0
    return float(max(0.0, 1.0 - var_r / var_sr))
