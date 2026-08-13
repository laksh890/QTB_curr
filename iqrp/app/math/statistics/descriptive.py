"""Descriptive statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from iqrp.app.math._array import as_array, as_vector


def mean(x: Any, *, axis: int | None = None) -> np.ndarray | float:
    arr = as_array(x)
    out = np.mean(arr, axis=axis)
    return float(out) if np.ndim(out) == 0 else np.asarray(out, dtype=np.float64)


def median(x: Any, *, axis: int | None = None) -> np.ndarray | float:
    arr = as_array(x)
    out = np.median(arr, axis=axis)
    return float(out) if np.ndim(out) == 0 else np.asarray(out, dtype=np.float64)


def mode(x: Any) -> float:
    v = as_vector(x)
    m = stats.mode(v, keepdims=False)
    return float(m.mode)


def variance(x: Any, *, ddof: int = 1, axis: int | None = None) -> np.ndarray | float:
    arr = as_array(x)
    out = np.var(arr, ddof=ddof, axis=axis)
    return float(out) if np.ndim(out) == 0 else np.asarray(out, dtype=np.float64)


def std(x: Any, *, ddof: int = 1, axis: int | None = None) -> np.ndarray | float:
    arr = as_array(x)
    out = np.std(arr, ddof=ddof, axis=axis)
    return float(out) if np.ndim(out) == 0 else np.asarray(out, dtype=np.float64)


def mad(x: Any, *, constant: float = 1.4826) -> float:
    """Median absolute deviation (scaled to match sigma under normality)."""
    v = as_vector(x)
    med = np.median(v)
    return float(constant * np.median(np.abs(v - med)))


def quantiles(x: Any, q: Any) -> np.ndarray:
    return np.asarray(np.quantile(as_vector(x), as_array(q)), dtype=np.float64)


def percentiles(x: Any, p: Any) -> np.ndarray:
    return np.asarray(np.percentile(as_vector(x), as_array(p)), dtype=np.float64)


def moment(x: Any, order: int, *, central: bool = True) -> float:
    v = as_vector(x)
    if central:
        v = v - np.mean(v)
    return float(np.mean(v**order))


def skewness(x: Any, *, bias: bool = False) -> float:
    return float(stats.skew(as_vector(x), bias=bias))


def kurtosis(x: Any, *, fisher: bool = True, bias: bool = False) -> float:
    return float(stats.kurtosis(as_vector(x), fisher=fisher, bias=bias))


def summarize(x: Any) -> dict[str, float]:
    v = as_vector(x)
    return {
        "n": float(v.size),
        "mean": float(mean(v)),
        "median": float(median(v)),
        "std": float(std(v)),
        "variance": float(variance(v)),
        "mad": mad(v),
        "skewness": skewness(v),
        "kurtosis": kurtosis(v),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "q25": float(np.quantile(v, 0.25)),
        "q75": float(np.quantile(v, 0.75)),
    }
