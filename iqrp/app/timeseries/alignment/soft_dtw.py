"""Soft-DTW divergence."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def soft_dtw(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    gamma: float = 1.0,
    normalize: bool = True,
) -> AnalysisResult:
    """Soft-DTW (Cuturi & Blondel) with log-sum-exp smoothing (FULL_SAMPLE)."""
    a = as_float_array(x)
    b = as_float_array(y)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n, m = a.size, b.size
    g = float(max(gamma, 1e-8))
    if n < 1 or m < 1:
        return AnalysisResult(
            method="soft_dtw",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="series are identical under soft alignment (soft-DTW=0)",
            alternative_hypothesis="series differ under soft-DTW alignment",
            parameters={"gamma": g, "normalize": normalize},
        )
    # pairwise squared distances
    C = (a[:, None] - b[None, :]) ** 2
    R = np.full((n + 2, m + 2), np.inf, dtype=np.float64)
    R[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r0 = -R[i - 1, j - 1] / g
            r1 = -R[i - 1, j] / g
            r2 = -R[i, j - 1] / g
            rmax = max(r0, r1, r2)
            softmin = -g * (rmax + np.log(np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)))
            R[i, j] = C[i - 1, j - 1] + softmin
    val = float(R[n, m])
    if normalize:
        val = val / (n + m)
    return AnalysisResult(
        method="soft_dtw",
        value=val,
        statistic=val,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="series are identical under soft alignment (soft-DTW=0)",
        alternative_hypothesis="series differ under soft-DTW alignment",
        parameters={"gamma": g, "normalize": normalize},
        metadata={"n": n, "m": m, "raw": float(R[n, m])},
    )
