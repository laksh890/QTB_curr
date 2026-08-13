"""Numerically stable elementary operations."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math._array import as_array


def logsumexp(a: Any, *, axis: int | None = None, keepdims: bool = False) -> np.ndarray | float:
    """Stable log-sum-exp: log(sum(exp(a)))."""
    x = as_array(a)
    if axis is None:
        m = np.max(x)
        if not np.isfinite(m):
            return float(m)
        return float(m + np.log(np.sum(np.exp(x - m))))
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return np.asarray(out, dtype=np.float64)


def stable_softmax(a: Any, *, axis: int = -1) -> np.ndarray:
    x = as_array(a)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return np.asarray(e / np.sum(e, axis=axis, keepdims=True), dtype=np.float64)


def safe_divide(
    numer: Any,
    denom: Any,
    *,
    fill: float = 0.0,
    eps: float = 0.0,
) -> np.ndarray:
    n = as_array(numer)
    d = as_array(denom)
    d_safe = np.where(np.abs(d) <= eps, np.nan, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = n / d_safe
    return np.where(np.isfinite(out), out, fill)


def clip_finite(
    a: Any,
    *,
    min_value: float = -1e300,
    max_value: float = 1e300,
) -> np.ndarray:
    x = as_array(a)
    x = np.where(np.isfinite(x), x, 0.0)
    return np.asarray(np.clip(x, min_value, max_value), dtype=np.float64)


def protect_overflow(a: Any, *, max_exp: float = 700.0) -> np.ndarray:
    """Clip values before exp to avoid overflow."""
    return np.asarray(np.clip(as_array(a), -max_exp, max_exp), dtype=np.float64)


def softplus(x: Any) -> np.ndarray:
    """Stable softplus: log(1 + exp(x))."""
    z = as_array(x)
    return np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
