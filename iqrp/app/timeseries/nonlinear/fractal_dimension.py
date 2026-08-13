"""Higuchi fractal dimension."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def higuchi_fd(
    x: np.ndarray | list[float],
    *,
    k_max: int = 10,
) -> AnalysisResult:
    """Higuchi fractal dimension of a 1-D series.

    Statistical descriptor only — not a guaranteed predictive signal.
    """
    y = as_float_array(x)
    finite = y[np.isfinite(y)]
    n = finite.size
    kmax = max(int(k_max), 2)
    if n < kmax * 2:
        return AnalysisResult(
            method="higuchi_fd",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="FD≈1 (smooth / low complexity path)",
            alternative_hypothesis="FD>1 (fractal / high complexity path)",
            parameters={"k_max": kmax},
        )
    lk = np.empty(kmax, dtype=np.float64)
    ks = np.arange(1, kmax + 1, dtype=np.float64)
    for ik, k in enumerate(range(1, kmax + 1)):
        lengths = []
        for m in range(1, k + 1):
            idx = np.arange(m - 1, n, k)
            if idx.size < 2:
                continue
            diffs = np.abs(np.diff(finite[idx]))
            norm = (n - 1) / (((idx.size - 1) * k) * k)
            lengths.append(float(np.sum(diffs) * norm))
        lk[ik] = float(np.mean(lengths)) if lengths else np.nan
    valid = np.isfinite(lk) & (lk > 0)
    if valid.sum() < 2:
        return AnalysisResult(
            method="higuchi_fd",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="FD≈1 (smooth / low complexity path)",
            alternative_hypothesis="FD>1 (fractal / high complexity path)",
            parameters={"k_max": kmax},
        )
    log_k = np.log(ks[valid])
    log_l = np.log(lk[valid])
    A = np.column_stack([np.ones(log_k.size), log_k])
    beta, *_ = np.linalg.lstsq(A, log_l, rcond=None)
    fd = float(-beta[1])
    fd = float(np.clip(fd, 1.0, 2.0))
    return AnalysisResult(
        method="higuchi_fd",
        value=fd,
        statistic=fd,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="FD≈1 (smooth / low complexity path)",
        alternative_hypothesis="FD>1 (fractal / high complexity path)",
        significant=fd > 1.2,
        parameters={"k_max": kmax},
        metadata={"n": n, "L_k": lk.tolist()},
    )
