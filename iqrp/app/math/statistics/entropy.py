"""Entropy and related information measures (core)."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math._array import as_array, as_vector
from iqrp.app.math.utils.numerical_stability import safe_divide


def _as_prob(p: Any) -> np.ndarray:
    arr = as_vector(p).astype(np.float64)
    arr = np.clip(arr, 0.0, None)
    s = arr.sum()
    if s <= 0:
        return np.full_like(arr, 1.0 / max(arr.size, 1))
    return arr / s


def entropy(p: Any, *, base: float = np.e) -> float:
    q = _as_prob(p)
    nz = q[q > 0]
    h = -np.sum(nz * np.log(nz))
    if base != np.e:
        h /= np.log(base)
    return float(h)


def cross_entropy(p: Any, q: Any, *, base: float = np.e) -> float:
    pp = _as_prob(p)
    qq = _as_prob(q)
    qq = np.clip(qq, 1e-300, None)
    ce = -np.sum(pp * np.log(qq))
    if base != np.e:
        ce /= np.log(base)
    return float(ce)


def kl_divergence(p: Any, q: Any, *, base: float = np.e) -> float:
    pp = _as_prob(p)
    qq = _as_prob(q)
    mask = pp > 0
    qq = np.clip(qq, 1e-300, None)
    kl = np.sum(pp[mask] * (np.log(pp[mask]) - np.log(qq[mask])))
    if base != np.e:
        kl /= np.log(base)
    return float(kl)


def js_divergence(p: Any, q: Any) -> float:
    pp = _as_prob(p)
    qq = _as_prob(q)
    m = 0.5 * (pp + qq)
    return 0.5 * kl_divergence(pp, m) + 0.5 * kl_divergence(qq, m)


def conditional_entropy(joint: Any) -> float:
    """H(X|Y) for joint table p(x,y) with rows=x, cols=y."""
    table = as_array(joint).astype(np.float64)
    table = table / table.sum()
    py = table.sum(axis=0, keepdims=True)
    px_y = safe_divide(table, py, fill=0.0)
    h = 0.0
    for j in range(table.shape[1]):
        col = px_y[:, j]
        w = float(py[0, j])
        if w > 0:
            h += w * entropy(col)
    return float(h)


def mutual_information(joint: Any) -> float:
    """I(X;Y) = H(X) + H(Y) - H(X,Y)."""
    table = as_array(joint).astype(np.float64)
    table = table / table.sum()
    px = table.sum(axis=1)
    py = table.sum(axis=0)
    return entropy(px) + entropy(py) - entropy(table.ravel())


def information_gain(parent_entropy: float, weighted_child_entropies: Any) -> float:
    return float(parent_entropy - np.sum(as_vector(weighted_child_entropies)))
