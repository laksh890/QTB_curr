"""Multi-period rebalancing schedules."""

from __future__ import annotations

from typing import Any

import numpy as np


def rebalance_schedule(
    n_periods: int,
    *,
    frequency: int = 1,
    threshold: float | None = None,
    calendar: list[int] | None = None,
    start: int = 0,
) -> dict[str, Any]:
    """
    Build a rebalancing schedule over ``n_periods`` decision points.

    - frequency: rebalance every k periods (1 = every period)
    - calendar: explicit period indices to rebalance
    - threshold: optional turnover trigger metadata (enforced by optimizer)
    """
    h = int(n_periods)
    if h < 0:
        raise ValueError("n_periods must be non-negative")
    if calendar is not None:
        times = sorted({int(t) for t in calendar if 0 <= int(t) < h})
    else:
        freq = max(int(frequency), 1)
        times = list(range(int(start) % freq, h, freq))
        if 0 not in times and h > 0:
            times = [0] + times
            times = sorted(set(times))
    flags = [i in set(times) for i in range(h)]
    return {
        "name": "rebalance_schedule",
        "n_periods": h,
        "rebalance_times": times,
        "flags": flags,
        "frequency": int(frequency),
        "threshold": None if threshold is None else float(threshold),
        "n_rebalances": len(times),
    }


def apply_drift(
    weights: Any,
    returns: Any,
) -> np.ndarray:
    """Drift portfolio weights by a single-period return vector."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    if w.size != r.size:
        raise ValueError("weights/returns size mismatch")
    grown = w * (1.0 + r)
    s = float(np.sum(grown))
    if abs(s) < 1e-14:
        return np.zeros_like(w)
    return grown / s


def turnover(w_from: Any, w_to: Any) -> float:
    a = np.asarray(w_from, dtype=np.float64).reshape(-1)
    b = np.asarray(w_to, dtype=np.float64).reshape(-1)
    return 0.5 * float(np.sum(np.abs(a - b)))
