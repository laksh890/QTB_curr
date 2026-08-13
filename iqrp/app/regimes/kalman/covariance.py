"""Covariance utilities for Kalman filters."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit as _njit  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False
    _njit = None
else:
    _HAS_NUMBA = True


def ensure_spd(matrix: Any, *, jitter: float = 1e-9) -> np.ndarray:
    """Project / regularize a matrix to be symmetric positive definite."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim == 0:
        return np.array([[max(float(m), jitter)]], dtype=np.float64)
    if m.ndim == 1:
        return np.diag(np.clip(m, jitter, None))
    m = 0.5 * (m + m.T)
    # eigenvalue clip
    try:
        vals, vecs = np.linalg.eigh(m)
        vals = np.clip(vals, jitter, None)
        return (vecs * vals) @ vecs.T
    except np.linalg.LinAlgError:
        return m + jitter * np.eye(m.shape[0])


def joseph_update(
    p_pred: np.ndarray,
    k: np.ndarray,
    h: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Joseph-form covariance update for numerical stability."""
    i = np.eye(p_pred.shape[0])
    ikh = i - k @ h
    return ensure_spd(ikh @ p_pred @ ikh.T + k @ r @ k.T)


def block_diag(*mats: np.ndarray) -> np.ndarray:
    mats_a = [np.atleast_2d(np.asarray(m, dtype=np.float64)) for m in mats]
    rows = sum(m.shape[0] for m in mats_a)
    cols = sum(m.shape[1] for m in mats_a)
    out = np.zeros((rows, cols), dtype=np.float64)
    r0 = c0 = 0
    for m in mats_a:
        r1, c1 = r0 + m.shape[0], c0 + m.shape[1]
        out[r0:r1, c0:c1] = m
        r0, c0 = r1, c1
    return out


def mahalanobis(innovation: np.ndarray, s: np.ndarray) -> float:
    v = np.asarray(innovation, dtype=np.float64).reshape(-1)
    s_spd = ensure_spd(s)
    try:
        sol = np.linalg.solve(s_spd, v)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(s_spd) @ v
    return float(np.sqrt(max(v @ sol, 0.0)))
