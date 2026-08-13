"""Z-score anomaly detection."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def zscore_anomalies(
    x: np.ndarray | list[float],
    *,
    threshold: float = 3.0,
    window: int | None = None,
) -> AnalysisResult:
    """Flag points with |z| > threshold (FULL_SAMPLE or ROLLING if window set)."""
    y = as_float_array(x)
    n = y.size
    thr = float(threshold)
    if n < 5 or not np.isfinite(y).any():
        return AnalysisResult(
            method="zscore_anomalies",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE if window is None else TemporalMode.ROLLING,
            null_hypothesis="observations arise from N(μ,σ²) without outliers",
            alternative_hypothesis="one or more points are anomalous (|z|>threshold)",
            parameters={"threshold": thr, "window": window},
        )
    if window is None:
        mu = float(np.nanmean(y))
        sd = float(np.nanstd(y, ddof=1))
        sd = sd if sd > 1e-12 else 1.0
        z = (y - mu) / sd
        mode = TemporalMode.FULL_SAMPLE
    else:
        w = max(int(window), 3)
        z = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            start = max(0, i - w + 1)
            chunk = y[start : i + 1]
            finite = chunk[np.isfinite(chunk)]
            if finite.size < 3:
                continue
            mu = float(np.mean(finite))
            sd = float(np.std(finite, ddof=1))
            sd = sd if sd > 1e-12 else 1.0
            if np.isfinite(y[i]):
                z[i] = (y[i] - mu) / sd
        mode = TemporalMode.ROLLING
    mask = np.isfinite(z) & (np.abs(z) > thr)
    indices = np.flatnonzero(mask).tolist()
    return AnalysisResult(
        method="zscore_anomalies",
        value={"indices": indices, "scores": z, "is_anomaly": mask},
        statistic=float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else np.nan,
        temporal_mode=mode,
        null_hypothesis="observations arise from N(μ,σ²) without outliers",
        alternative_hypothesis="one or more points are anomalous (|z|>threshold)",
        significant=len(indices) > 0,
        parameters={"threshold": thr, "window": window},
        metadata={"n": n, "n_anomalies": len(indices)},
    )
