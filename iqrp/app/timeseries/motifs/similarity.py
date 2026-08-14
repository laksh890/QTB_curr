"""Subsequence similarity utilities."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def subsequence_distance(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    *,
    z_normalize: bool = True,
) -> AnalysisResult:
    """Z-normalized Euclidean distance between two equal-length subsequences."""
    x = as_float_array(a)
    y = as_float_array(b)
    n = min(x.size, y.size)
    if n < 2:
        return AnalysisResult(
            method="subsequence_distance",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="subsequences are identical (distance=0)",
            alternative_hypothesis="subsequences differ",
            parameters={"z_normalize": z_normalize},
        )
    x, y = x[:n], y[:n]
    if z_normalize:
        x = _znorm(x)
        y = _znorm(y)
    dist = float(np.sqrt(np.sum((x - y) ** 2)))
    return AnalysisResult(
        method="subsequence_distance",
        value=dist,
        statistic=dist,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="subsequences are identical (distance=0)",
        alternative_hypothesis="subsequences differ",
        parameters={"z_normalize": z_normalize, "length": n},
    )


def nearest_neighbors(
    query: np.ndarray | list[float],
    series: np.ndarray | list[float],
    *,
    top_k: int = 5,
    z_normalize: bool = True,
) -> AnalysisResult:
    """Find top-k nearest subsequences of ``query`` length in ``series`` (FULL_SAMPLE)."""
    q = as_float_array(query)
    y = as_float_array(series)
    m = q.size
    k = max(int(top_k), 1)
    if m < 2 or y.size < m:
        return AnalysisResult(
            method="nearest_neighbors",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis=None,
            alternative_hypothesis=None,
            parameters={"top_k": k, "z_normalize": z_normalize},
        )
    qn = _znorm(q) if z_normalize else q
    n_sub = y.size - m + 1
    dists = np.empty(n_sub, dtype=np.float64)
    for i in range(n_sub):
        sub = y[i : i + m]
        sn = _znorm(sub) if z_normalize else sub
        dists[i] = float(np.sqrt(np.sum((qn - sn) ** 2)))
    # exclude trivial self-match if query is taken from series (distance~0)
    order = np.argsort(dists)
    neighbors: list[dict] = []
    for i in order:
        i = int(i)
        neighbors.append(
            {"index": i, "distance": float(dists[i]), "subsequence": y[i : i + m].tolist()}
        )
        if len(neighbors) >= k:
            break
    return AnalysisResult(
        method="nearest_neighbors",
        value=neighbors,
        statistic=neighbors[0]["distance"] if neighbors else np.nan,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"top_k": k, "z_normalize": z_normalize, "query_length": m},
        metadata={"n_subsequences": n_sub},
    )


def _znorm(v: np.ndarray) -> np.ndarray:
    sd = float(np.std(v))
    if sd < 1e-12:
        return v - np.mean(v)
    return (v - np.mean(v)) / sd
