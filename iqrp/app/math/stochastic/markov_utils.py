"""Markov-chain mathematical utilities (not financial models)."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from iqrp.app.math._array import as_matrix, as_vector
from iqrp.app.math.matrices.matrix import normalize_rows


def is_stochastic(matrix: Any, *, tol: float = 1e-8) -> bool:
    p = as_matrix(matrix)
    if np.any(p < -tol):
        return False
    return bool(np.allclose(p.sum(axis=1), 1.0, atol=tol))


def stationary_distribution(transition: Any, *, tol: float = 1e-12) -> np.ndarray:
    """Solve π P = π, π 1 = 1."""
    p = as_matrix(transition).astype(np.float64)
    k = p.shape[0]
    a = np.vstack([p.T - np.eye(k), np.ones(k)])
    b = np.zeros(k + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(a, b, rcond=None)
    pi = np.clip(pi, 0.0, None)
    s = pi.sum()
    return pi / s if s > 0 else np.full(k, 1.0 / k)


def n_step_transition(transition: Any, n: int) -> np.ndarray:
    p = as_matrix(transition).astype(np.float64)
    return np.linalg.matrix_power(p, max(0, int(n)))


def simulate_markov(
    transition: Any,
    n_steps: int,
    *,
    initial: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    p = normalize_rows(as_matrix(transition))
    k = p.shape[0]
    states = np.empty(n_steps, dtype=np.int64)
    states[0] = int(initial if initial is not None else rng.integers(0, k))
    for t in range(1, n_steps):
        states[t] = int(rng.choice(k, p=p[states[t - 1]]))
    return states


def empirical_transition(states: Any, n_states: int | None = None) -> np.ndarray:
    s = as_vector(states).astype(np.int64)
    k = int(n_states if n_states is not None else (s.max() + 1 if s.size else 0))
    tm = np.zeros((k, k), dtype=np.float64)
    for a, b in pairwise(s):
        if 0 <= a < k and 0 <= b < k:
            tm[a, b] += 1.0
    return normalize_rows(tm)


def mixing_time_bound(transition: Any) -> float:
    """Crude bound using spectral gap of reversible chains: 1 / (1 - |λ2|)."""
    p = as_matrix(transition)
    vals = np.sort(np.abs(np.linalg.eigvals(p)))[::-1]
    if len(vals) < 2:
        return float("inf")
    gap = 1.0 - float(vals[1])
    return float(1.0 / max(gap, 1e-12))
