"""Classical additive/multiplicative seasonal decomposition."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import DecompositionResult, TemporalMode, as_float_array


def classical_decompose(
    x: np.ndarray | list[float],
    *,
    period: int = 24,
    model: str = "additive",
) -> DecompositionResult:
    arr = as_float_array(x)
    p = max(int(period), 2)
    n = arr.size
    # trend via centered moving average
    trend = _centered_ma(arr, p)
    if model == "multiplicative":
        safe = np.where(np.abs(trend) > 1e-12, trend, np.nan)
        detrended = arr / safe
    else:
        detrended = arr - trend
    seasonal = _seasonal_means(detrended, p)
    if model == "multiplicative":
        residual = np.where(np.abs(trend * seasonal) > 1e-12, arr / (trend * seasonal), np.nan)
    else:
        residual = arr - trend - seasonal
    return DecompositionResult(
        method="classical",
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        observed=arr,
        model=model,
        parameters={"period": p},
        temporal_mode=TemporalMode.FULL_SAMPLE,
        metadata={"n": n},
    )


def _centered_ma(arr: np.ndarray, period: int) -> np.ndarray:
    n = arr.size
    out = np.full(n, np.nan)
    if period % 2 == 0:
        w = period
        kernel = np.ones(w + 1) / (w + 1)
        kernel[0] = kernel[-1] = 0.5 / w
        half = w // 2
    else:
        kernel = np.ones(period) / period
        half = period // 2
    for i in range(half, n - half):
        if period % 2 == 0:
            chunk = arr[i - half : i + half + 1]
            if chunk.size == kernel.size:
                out[i] = float(np.nansum(chunk * kernel))
        else:
            chunk = arr[i - half : i + half + 1]
            out[i] = float(np.nanmean(chunk))
    # fill edges with nearest
    valid = np.where(np.isfinite(out))[0]
    if valid.size:
        out[: valid[0]] = out[valid[0]]
        out[valid[-1] + 1 :] = out[valid[-1]]
    return out


def _seasonal_means(detrended: np.ndarray, period: int) -> np.ndarray:
    n = detrended.size
    seas = np.zeros(period)
    counts = np.zeros(period)
    for i, v in enumerate(detrended):
        if np.isfinite(v):
            seas[i % period] += v
            counts[i % period] += 1
    counts = np.maximum(counts, 1)
    seas = seas / counts
    seas = seas - np.mean(seas)
    return np.tile(seas, int(np.ceil(n / period)))[:n]
