"""Stationary distribution and chain structural diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.math.stochastic.markov_utils import mixing_time_bound, stationary_distribution


class StationaryAnalyzer:
    """Steady-state probabilities and structural properties of ``P``."""

    def analyze(self, transition: Any) -> dict[str, Any]:
        p = normalize_rows(np.asarray(transition, dtype=np.float64))
        pi = stationary_distribution(p)
        return {
            "stationary_distribution": pi,
            "steady_state_probabilities": pi.copy(),
            "is_irreducible": is_irreducible(p),
            "is_aperiodic": is_aperiodic(p),
            "is_ergodic": is_ergodic(p),
            "period": estimate_period(p),
            "mixing_time": float(mixing_time_bound(p)),
            "spectral_gap": spectral_gap(p),
        }

    def stationary_distribution(self, transition: Any) -> np.ndarray:
        return stationary_distribution(transition)


def spectral_gap(transition: Any) -> float:
    p = np.asarray(transition, dtype=np.float64)
    vals = np.sort(np.abs(np.linalg.eigvals(p)))[::-1]
    if len(vals) < 2:
        return 0.0
    return float(max(0.0, 1.0 - float(vals[1])))


def is_irreducible(transition: Any, *, tol: float = 1e-12) -> bool:
    """Graph strongly connected via ``(I + A)^{K}`` reachability on support of ``P``."""
    p = np.asarray(transition, dtype=np.float64)
    k = p.shape[0]
    adj = (p > tol).astype(np.float64)
    reach = np.eye(k, dtype=np.float64)
    eye_adj = np.eye(k) + adj
    for _ in range(k):
        reach = reach @ eye_adj
    return bool(np.all(reach > tol))


def is_aperiodic(transition: Any, *, tol: float = 1e-12) -> bool:
    """Heuristic: gcd of self-loop / cycle lengths ≈ 1 via diagonal of powers."""
    return estimate_period(transition, tol=tol) == 1


def estimate_period(transition: Any, *, tol: float = 1e-12) -> int:
    """Approximate period as gcd of times ``t`` where ``(P^t)_ii > 0`` for some ``i``."""
    p = np.asarray(transition, dtype=np.float64)
    k = p.shape[0]
    if k == 0:
        return 1
    # If any self-loop, chain is aperiodic on that communicating class
    if np.any(np.diag(p) > tol):
        return 1
    power = np.eye(k, dtype=np.float64)
    times: list[int] = []
    for t in range(1, 2 * k + 1):
        power = power @ p
        if np.any(np.diag(power) > tol):
            times.append(t)
            if len(times) >= 4:
                break
    if not times:
        return 1
    g = times[0]
    for t in times[1:]:
        g = int(np.gcd(g, t))
    return max(int(g), 1)


def is_ergodic(transition: Any, *, tol: float = 1e-12) -> bool:
    return is_irreducible(transition, tol=tol) and is_aperiodic(transition, tol=tol)
