"""CUSUM change-point detection for mean shifts."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import ChangePointResult, TemporalMode, as_float_array


def cusum_detect(
    x: np.ndarray | list[float],
    *,
    threshold: float | None = None,
    drift: float = 0.0,
    min_size: int = 5,
) -> ChangePointResult:
    """Offline two-sided CUSUM for mean shifts (FULL_SAMPLE).

    Uses the classic cumulative-sum of standardized deviations from the
    full-sample mean; change-points are local extrema of |S_t| exceeding
    a threshold (default: 1.5 * sqrt(n) after standardization).
    """
    y = as_float_array(x)
    mask = np.isfinite(y)
    if int(mask.sum()) < max(min_size * 2, 10):
        return ChangePointResult(
            method="cusum",
            indices=[],
            scores=None,
            kind="mean",
            parameters={"threshold": threshold, "drift": drift, "min_size": min_size},
            temporal_mode=TemporalMode.FULL_SAMPLE,
            metadata={"status": "insufficient_data", "n": int(y.size)},
        )
    # work on finite values but map indices back
    idx_map = np.flatnonzero(mask)
    z = y[mask]
    n = z.size
    mu = float(np.mean(z))
    sd = float(np.std(z, ddof=1)) if n > 1 else 1.0
    sd = sd if sd > 1e-12 else 1.0
    standardized = (z - mu) / sd
    s = np.cumsum(standardized - drift)
    s = s - np.linspace(0.0, s[-1], n)  # bridge to remove end-point bias
    scores_full = np.full(y.size, np.nan, dtype=np.float64)
    scores_full[idx_map] = s

    thr = float(threshold) if threshold is not None else 1.5 * np.sqrt(n)
    abs_s = np.abs(s)
    candidates: list[int] = []
    i = min_size
    while i < n - min_size:
        if abs_s[i] >= thr and abs_s[i] >= abs_s[i - 1] and abs_s[i] >= abs_s[i + 1]:
            # local peak above threshold
            candidates.append(int(idx_map[i]))
            i += min_size  # enforce separation
        else:
            i += 1

    return ChangePointResult(
        method="cusum",
        indices=candidates,
        scores=scores_full,
        kind="mean",
        parameters={"threshold": thr, "drift": drift, "min_size": min_size},
        temporal_mode=TemporalMode.FULL_SAMPLE,
        metadata={"n": int(y.size), "n_finite": n, "mean": mu, "std": sd},
    )
