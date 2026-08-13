"""Trend extraction helpers."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array
from iqrp.app.timeseries.decomposition.stl import stl_decompose


def extract_trend(
    x: np.ndarray | list[float],
    *,
    period: int = 24,
    method: str = "stl",
) -> AnalysisResult:
    arr = as_float_array(x)
    if method == "hp":
        trend = _hp_filter(arr, lamb=1600.0)
    else:
        trend = stl_decompose(arr, period=period).trend
    strength = _trend_strength(trend, arr - trend)
    return AnalysisResult(
        method=f"trend.{method}",
        value=trend,
        parameters={"period": period},
        temporal_mode=TemporalMode.FULL_SAMPLE,
        metadata={"trend_strength": strength},
    )


def trend_strength(trend: np.ndarray, detrended: np.ndarray) -> float:
    return _trend_strength(trend, detrended)


def _trend_strength(trend: np.ndarray, residual: np.ndarray) -> float:
    t = as_float_array(trend)
    r = as_float_array(residual)
    var_r = float(np.nanvar(r))
    var_tr = float(np.nanvar(t + r))
    if var_tr <= 1e-15:
        return 0.0
    return float(max(0.0, 1.0 - var_r / var_tr))


def _hp_filter(y: np.ndarray, lamb: float = 1600.0) -> np.ndarray:
    n = y.size
    if n < 4:
        return y.copy()
    # sparse second-difference smoother via dense solve for modest n
    I = np.eye(n)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    A = I + lamb * (D.T @ D)
    return np.linalg.solve(A, y)
