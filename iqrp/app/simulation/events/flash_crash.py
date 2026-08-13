"""Flash-crash event injector."""

from __future__ import annotations

import numpy as np


def inject_flash_crashes(
    prices: np.ndarray,
    *,
    probability: float,
    rng: np.random.Generator,
    severity: float = 0.08,
    recovery_bars: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply sudden drops with partial recovery. Returns (prices, mask)."""
    out = np.asarray(prices, dtype=np.float64).copy()
    n = len(out)
    mask = np.zeros(n, dtype=np.bool_)
    if n < 2 or probability <= 0:
        return out, mask
    for t in range(1, n - recovery_bars - 1):
        if rng.random() < probability:
            mask[t] = True
            crash = 1.0 - abs(severity) * (0.5 + rng.random())
            out[t] = max(1e-8, out[t] * crash)
            # Mean-revert recovery toward pre-crash path
            target = out[t - 1]
            for k in range(1, recovery_bars + 1):
                w = k / (recovery_bars + 1)
                out[t + k] = (1 - w) * out[t + k] + w * target * (0.97 + 0.03 * rng.random())
                out[t + k] = max(1e-8, out[t + k])
    return out, mask
