"""Covariance estimators."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math._array import as_matrix, as_vector


def covariance(x: Any, y: Any | None = None, *, ddof: int = 1) -> np.ndarray | float:
    if y is None:
        mat = as_matrix(x)
        return np.cov(mat, rowvar=False, ddof=ddof)
    return float(np.cov(as_vector(x), as_vector(y), ddof=ddof)[0, 1])


def covariance_matrix(x: Any, *, ddof: int = 1) -> np.ndarray:
    mat = as_matrix(x)
    return np.asarray(np.cov(mat, rowvar=False, ddof=ddof), dtype=np.float64)


def correlation_from_covariance(cov: Any) -> np.ndarray:
    c = np.asarray(cov, dtype=np.float64)
    d = np.sqrt(np.clip(np.diag(c), 1e-300, None))
    return c / np.outer(d, d)


def shrunk_covariance(x: Any, *, shrinkage: float = 0.1) -> np.ndarray:
    """Ledoit-Wolf-style simple shrinkage toward diagonal."""
    emp = covariance_matrix(x)
    target = np.diag(np.diag(emp))
    alpha = float(np.clip(shrinkage, 0.0, 1.0))
    return (1.0 - alpha) * emp + alpha * target


def rolling_covariance(x: Any, y: Any, window: int) -> np.ndarray:
    a = as_vector(x)
    b = as_vector(y)
    n = min(len(a), len(b))
    out = np.full(n, np.nan, dtype=np.float64)
    w = max(2, int(window))
    for i in range(w - 1, n):
        out[i] = float(np.cov(a[i - w + 1 : i + 1], b[i - w + 1 : i + 1])[0, 1])
    return out
