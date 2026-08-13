"""Exchange downtime / stale quote injector."""

from __future__ import annotations

import numpy as np


def inject_exchange_outages(
    prices: np.ndarray,
    volumes: np.ndarray,
    *,
    probability: float,
    rng: np.random.Generator,
    max_duration: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Freeze prices and zero volume during outages.

    Returns (prices, volumes, mask).
    """
    out_p = np.asarray(prices, dtype=np.float64).copy()
    out_v = np.asarray(volumes, dtype=np.float64).copy()
    n = len(out_p)
    mask = np.zeros(n, dtype=np.bool_)
    t = 1
    while t < n:
        if rng.random() < probability:
            dur = int(rng.integers(1, max_duration + 1))
            end = min(n, t + dur)
            mask[t:end] = True
            out_p[t:end] = out_p[t - 1]
            out_v[t:end] = 0.0
            t = end
        else:
            t += 1
    return out_p, out_v, mask
