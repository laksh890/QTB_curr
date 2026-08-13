"""Correlation measures."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from iqrp.app.math._array import as_matrix, as_vector
from iqrp.app.math.statistics.covariance import correlation_from_covariance, covariance_matrix


def pearson(x: Any, y: Any) -> float:
    r, _ = stats.pearsonr(as_vector(x), as_vector(y))
    return float(r)


def spearman(x: Any, y: Any) -> float:
    r, _ = stats.spearmanr(as_vector(x), as_vector(y))
    return float(r)


def kendall(x: Any, y: Any) -> float:
    r, _ = stats.kendalltau(as_vector(x), as_vector(y))
    return float(r)


def correlation_matrix(x: Any) -> np.ndarray:
    return correlation_from_covariance(covariance_matrix(x))


def distance_correlation(x: Any, y: Any) -> float:
    """Szekely distance correlation in [0, 1]."""
    a = as_vector(x).astype(np.float64)
    b = as_vector(y).astype(np.float64)
    n = len(a)
    if n < 2:
        return float("nan")
    ax = np.abs(a[:, None] - a[None, :])
    ay = np.abs(b[:, None] - b[None, :])
    a_cent = ax - ax.mean(axis=0) - ax.mean(axis=1)[:, None] + ax.mean()
    b_cent = ay - ay.mean(axis=0) - ay.mean(axis=1)[:, None] + ay.mean()
    dcov2 = float(np.mean(a_cent * b_cent))
    dvarx = float(np.mean(a_cent * a_cent))
    dvary = float(np.mean(b_cent * b_cent))
    if dvarx <= 0 or dvary <= 0:
        return 0.0
    return float(np.sqrt(dcov2) / np.sqrt(np.sqrt(dvarx * dvary)))


def cross_correlation(x: Any, y: Any, *, max_lag: int = 20) -> np.ndarray:
    a = as_vector(x) - np.mean(as_vector(x))
    b = as_vector(y) - np.mean(as_vector(y))
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-15
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.empty(len(lags), dtype=np.float64)
    for i, lag in enumerate(lags):
        if lag < 0:
            out[i] = float(np.dot(a[-lag:], b[: len(b) + lag]) / denom)
        elif lag > 0:
            out[i] = float(np.dot(a[:-lag], b[lag:]) / denom)
        else:
            out[i] = float(np.dot(a, b) / denom)
    return out


def rolling_correlation(x: Any, y: Any, window: int) -> np.ndarray:
    a = as_vector(x)
    b = as_vector(y)
    n = min(len(a), len(b))
    out = np.full(n, np.nan, dtype=np.float64)
    w = max(2, int(window))
    for i in range(w - 1, n):
        out[i] = pearson(a[i - w + 1 : i + 1], b[i - w + 1 : i + 1])
    return out


def pairwise_correlations(x: Any, method: str = "pearson") -> np.ndarray:
    mat = as_matrix(x)
    k = mat.shape[1]
    out = np.eye(k, dtype=np.float64)
    fn = {"pearson": pearson, "spearman": spearman, "kendall": kendall}[method]
    for i in range(k):
        for j in range(i + 1, k):
            r = fn(mat[:, i], mat[:, j])
            out[i, j] = out[j, i] = r
    return out
