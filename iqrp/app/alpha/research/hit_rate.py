"""Hit-rate metrics for directional predictive research.

CRITICAL:
- Hit rate is a triage metric, not proof of alpha.
- Statistical significance alone ≠ alpha.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_hit_rate(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    *,
    signal_threshold: float = 0.0,
    return_threshold: float = 0.0,
) -> float:
    """Fraction of periods where signal direction matches forward return direction."""
    x = np.asarray(signal, dtype=np.float64)
    y = np.asarray(forward_returns, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError("length mismatch")
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return float("nan")
    sx = np.sign(x[m] - signal_threshold)
    sy = np.sign(y[m] - return_threshold)
    # Ignore exact zeros on either side
    valid = (sx != 0) & (sy != 0)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(sx[valid] == sy[valid]))


def rolling_hit_rate(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    *,
    window: int = 60,
    step: int = 1,
    min_obs: int = 20,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    y = np.asarray(forward_returns, dtype=np.float64)
    out: list[float] = []
    for start in range(0, max(0, len(x) - window + 1), step):
        sl = slice(start, start + window)
        if np.isfinite(x[sl]).sum() < min_obs:
            out.append(float("nan"))
            continue
        out.append(compute_hit_rate(x[sl], y[sl]))
    return np.asarray(out, dtype=np.float64)


def hit_rate_summary(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    *,
    window: int = 60,
    step: int = 20,
) -> dict[str, Any]:
    hr = compute_hit_rate(signal, forward_returns)
    roll = rolling_hit_rate(signal, forward_returns, window=window, step=step)
    finite = roll[np.isfinite(roll)]
    return {
        "hit_rate": hr,
        "rolling_hit_rate_mean": float(np.mean(finite)) if finite.size else float("nan"),
        "rolling_hit_rate_std": float(np.std(finite)) if finite.size else float("nan"),
        "n_rolling": int(finite.size),
        "disclaimer": "Hit rate ≠ alpha. Statistical significance alone ≠ alpha.",
    }
