"""STL-style seasonal-trend decomposition (LOESS-lite via local polynomials)."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import DecompositionResult, TemporalMode, as_float_array
from iqrp.app.timeseries.decomposition.classical import classical_decompose


def stl_decompose(
    x: np.ndarray | list[float],
    *,
    period: int = 24,
    seasonal_window: int | None = None,
    trend_window: int | None = None,
    robust: bool = False,
    n_iter: int = 2,
) -> DecompositionResult:
    arr = as_float_array(x)
    p = max(int(period), 2)
    # initialize via classical
    init = classical_decompose(arr, period=p, model="additive")
    seasonal = init.seasonal.copy()
    trend = init.trend.copy()
    sw = seasonal_window or max(7, p + (1 - p % 2))
    tw = trend_window or max(p * 2 + 1, 15)
    if tw % 2 == 0:
        tw += 1
    if sw % 2 == 0:
        sw += 1
    weights = np.ones(arr.size)
    for _ in range(max(n_iter, 1)):
        detrended = arr - trend
        seasonal = _loess_seasonal(detrended, p, sw, weights)
        deseason = arr - seasonal
        trend = _loess_smooth(deseason, tw, weights)
        resid = arr - trend - seasonal
        if robust:
            weights = _robust_weights(resid)
    residual = arr - trend - seasonal
    return DecompositionResult(
        method="stl",
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        observed=arr,
        model="additive",
        parameters={"period": p, "seasonal_window": sw, "trend_window": tw, "robust": robust},
        temporal_mode=TemporalMode.FULL_SAMPLE,
        metadata={"robust": robust},
    )


def _loess_smooth(y: np.ndarray, window: int, weights: np.ndarray) -> np.ndarray:
    n = y.size
    half = window // 2
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        idx = np.arange(lo, hi)
        d = np.abs(idx - i) / max(half, 1)
        w = (1 - d**3) ** 3
        w = w * weights[lo:hi]
        w = np.where(np.isfinite(y[lo:hi]), w, 0.0)
        if w.sum() <= 1e-12:
            out[i] = y[i] if np.isfinite(y[i]) else 0.0
        else:
            out[i] = float(np.sum(w * y[lo:hi]) / w.sum())
    return out


def _loess_seasonal(y: np.ndarray, period: int, window: int, weights: np.ndarray) -> np.ndarray:
    # smooth each seasonal subseries then cycle
    n = y.size
    out = np.zeros(n)
    for r in range(period):
        idx = np.arange(r, n, period)
        if idx.size == 0:
            continue
        sub = y[idx]
        wsub = weights[idx]
        sm = _loess_smooth(sub, min(window, max(idx.size | 1, 3) | 1), wsub)
        out[idx] = sm
    # center seasonal
    means = np.array([np.nanmean(out[r::period]) for r in range(period)])
    means = means - np.nanmean(means)
    for r in range(period):
        out[r::period] = means[r]
    return out


def _robust_weights(resid: np.ndarray) -> np.ndarray:
    abs_r = np.abs(resid)
    mad = np.nanmedian(abs_r)
    c = 6.0 * (mad if mad > 1e-12 else 1.0)
    u = abs_r / c
    w = np.where(u < 1.0, (1 - u**2) ** 2, 0.0)
    return np.where(np.isfinite(w), w, 0.0)
