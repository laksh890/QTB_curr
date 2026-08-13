"""News-shock and gap-open event injectors."""

from __future__ import annotations

import numpy as np


def inject_news_shocks(
    prices: np.ndarray,
    *,
    probability: float,
    rng: np.random.Generator,
    shock_std: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(prices, dtype=np.float64).copy()
    mask = np.zeros(len(out), dtype=np.bool_)
    for t in range(1, len(out)):
        if rng.random() < probability:
            mask[t] = True
            shock = rng.normal(0.0, shock_std)
            out[t:] *= np.exp(shock)
    return out, mask


def inject_gap_opens(
    prices: np.ndarray,
    *,
    probability: float,
    rng: np.random.Generator,
    gap_std: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Overnight / session gap discontinuities."""
    out = np.asarray(prices, dtype=np.float64).copy()
    mask = np.zeros(len(out), dtype=np.bool_)
    for t in range(1, len(out)):
        if rng.random() < probability:
            mask[t] = True
            gap = rng.normal(0.0, gap_std)
            out[t:] *= np.exp(gap)
    return out, mask


def inject_momentum_bursts(
    prices: np.ndarray,
    *,
    probability: float,
    rng: np.random.Generator,
    burst_len: int = 5,
    drift: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(prices, dtype=np.float64).copy()
    n = len(out)
    mask = np.zeros(n, dtype=np.bool_)
    t = 1
    while t < n - burst_len:
        if rng.random() < probability:
            sign = 1.0 if rng.random() > 0.5 else -1.0
            for k in range(burst_len):
                mask[t + k] = True
                out[t + k :] *= np.exp(sign * abs(drift) * (0.5 + rng.random()))
            t += burst_len
        else:
            t += 1
    return out, mask


def inject_slow_trends(
    prices: np.ndarray,
    *,
    rng: np.random.Generator,
    strength: float = 0.0002,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth cumulative drift overlay (always-on mild trend component)."""
    out = np.asarray(prices, dtype=np.float64).copy()
    n = len(out)
    sign = 1.0 if rng.random() > 0.5 else -1.0
    trend = np.exp(sign * strength * np.arange(n, dtype=np.float64))
    out = out * trend
    mask = np.ones(n, dtype=np.bool_)
    return out, mask
