"""Dynamic Time Warping distance and path."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def dtw_distance(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    window: int | None = None,
    normalize: bool = True,
) -> AnalysisResult:
    """Classic DTW distance with optional Sakoe–Chiba band (FULL_SAMPLE)."""
    a = as_float_array(x)
    b = as_float_array(y)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n, m = a.size, b.size
    if n < 1 or m < 1:
        return AnalysisResult(
            method="dtw_distance",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="series are identical under time warping (DTW=0)",
            alternative_hypothesis="series differ under optimal alignment",
            parameters={"window": window, "normalize": normalize},
        )
    dist, _ = _dtw_cost(a, b, window=window)
    value = dist / (n + m) if normalize else dist
    return AnalysisResult(
        method="dtw_distance",
        value=float(value),
        statistic=float(value),
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="series are identical under time warping (DTW=0)",
        alternative_hypothesis="series differ under optimal alignment",
        parameters={"window": window, "normalize": normalize},
        metadata={"raw_distance": float(dist), "n": n, "m": m},
    )


def dtw_path(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    window: int | None = None,
) -> AnalysisResult:
    """Return DTW distance and optimal warping path as (i, j) pairs (FULL_SAMPLE)."""
    a = as_float_array(x)
    b = as_float_array(y)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n, m = a.size, b.size
    if n < 1 or m < 1:
        return AnalysisResult(
            method="dtw_path",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="series are identical under time warping (DTW=0)",
            alternative_hypothesis="series differ under optimal alignment",
            parameters={"window": window},
        )
    dist, path = _dtw_cost(a, b, window=window, return_path=True)
    return AnalysisResult(
        method="dtw_path",
        value={"distance": float(dist), "path": path},
        statistic=float(dist),
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="series are identical under time warping (DTW=0)",
        alternative_hypothesis="series differ under optimal alignment",
        parameters={"window": window},
        metadata={"n": n, "m": m, "path_length": len(path)},
    )


def _dtw_cost(
    a: np.ndarray,
    b: np.ndarray,
    *,
    window: int | None = None,
    return_path: bool = False,
) -> tuple[float, list[tuple[int, int]]]:
    n, m = a.size, b.size
    band = int(window) if window is not None else max(n, m)
    band = max(band, abs(n - m))
    INF = 1e300
    D = np.full((n + 1, m + 1), INF, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    dist = float(np.sqrt(D[n, m])) if D[n, m] < INF / 2 else float("inf")
    if not return_path:
        return dist, []
    # backtrack
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (D[i - 1, j - 1], i - 1, j - 1),
            (D[i - 1, j], i - 1, j),
            (D[i, j - 1], i, j - 1),
        ]
        _, i, j = min(candidates, key=lambda t: t[0])
    path.reverse()
    return dist, path
