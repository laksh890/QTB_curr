"""Robust anomaly detection (MAD / robust z-score)."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def robust_zscore_anomalies(
    x: np.ndarray | list[float],
    *,
    threshold: float = 3.5,
    window: int | None = None,
) -> AnalysisResult:
    """Modified z-score using median and MAD (FULL_SAMPLE or ROLLING)."""
    y = as_float_array(x)
    n = y.size
    thr = float(threshold)
    if n < 5 or not np.isfinite(y).any():
        return AnalysisResult(
            method="robust_zscore_anomalies",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE if window is None else TemporalMode.ROLLING,
            null_hypothesis="no robust outliers (|modified-z|≤threshold)",
            alternative_hypothesis="one or more robust outliers present",
            parameters={"threshold": thr, "window": window},
        )
    if window is None:
        med = float(np.nanmedian(y))
        mad = float(np.nanmedian(np.abs(y - med)))
        scale = 1.4826 * mad if mad > 1e-12 else 1.0
        z = (y - med) / scale
        mode = TemporalMode.FULL_SAMPLE
    else:
        w = max(int(window), 3)
        z = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            start = max(0, i - w + 1)
            chunk = y[start : i + 1]
            finite = chunk[np.isfinite(chunk)]
            if finite.size < 3 or not np.isfinite(y[i]):
                continue
            med = float(np.median(finite))
            mad = float(np.median(np.abs(finite - med)))
            scale = 1.4826 * mad if mad > 1e-12 else 1.0
            z[i] = (y[i] - med) / scale
        mode = TemporalMode.ROLLING
    mask = np.isfinite(z) & (np.abs(z) > thr)
    indices = np.flatnonzero(mask).tolist()
    return AnalysisResult(
        method="robust_zscore_anomalies",
        value={"indices": indices, "scores": z, "is_anomaly": mask},
        statistic=float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else np.nan,
        temporal_mode=mode,
        null_hypothesis="no robust outliers (|modified-z|≤threshold)",
        alternative_hypothesis="one or more robust outliers present",
        significant=len(indices) > 0,
        parameters={"threshold": thr, "window": window},
        metadata={"n": n, "n_anomalies": len(indices)},
    )


def mad_anomalies(
    x: np.ndarray | list[float],
    *,
    threshold: float = 3.5,
) -> AnalysisResult:
    """Alias for full-sample MAD-based robust z-score anomalies."""
    res = robust_zscore_anomalies(x, threshold=threshold, window=None)
    return AnalysisResult(
        method="mad_anomalies",
        value=res.value,
        statistic=res.statistic,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis=res.null_hypothesis,
        alternative_hypothesis=res.alternative_hypothesis,
        significant=res.significant,
        parameters=dict(res.parameters),
        metadata=dict(res.metadata),
    )
