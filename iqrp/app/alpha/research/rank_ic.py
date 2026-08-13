"""Rank information coefficient for alpha research.

Imports ``rank_information_coefficient`` from features research numerics.

CRITICAL: Rank IC is a research association metric — not approval evidence.
Statistical significance alone ≠ alpha.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.features.research._numeric import (
    rank_information_coefficient,
    safe_nanmean,
)


def compute_rank_ic(signal: np.ndarray, forward_returns: np.ndarray) -> float:
    """Spearman / rank IC between signal and forward returns."""
    x = np.asarray(signal, dtype=np.float64)
    y = np.asarray(forward_returns, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError(f"length mismatch: signal={len(x)} forward={len(y)}")
    return float(rank_information_coefficient(x, y))


def rolling_rank_ic(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    *,
    window: int = 60,
    step: int = 1,
    min_obs: int = 20,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    y = np.asarray(forward_returns, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError("length mismatch")
    out: list[float] = []
    for start in range(0, max(0, len(x) - window + 1), step):
        sl = slice(start, start + window)
        xs, ys = x[sl], y[sl]
        if np.isfinite(xs).sum() < min_obs:
            out.append(float("nan"))
            continue
        out.append(compute_rank_ic(xs, ys))
    return np.asarray(out, dtype=np.float64)


def rank_ic_summary(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    *,
    window: int = 60,
    step: int = 20,
) -> dict[str, Any]:
    ric = compute_rank_ic(signal, forward_returns)
    roll = rolling_rank_ic(signal, forward_returns, window=window, step=step)
    finite = roll[np.isfinite(roll)]
    return {
        "rank_ic": ric,
        "rolling_rank_ic_mean": safe_nanmean(roll),
        "rolling_rank_ic_std": float(np.std(finite)) if finite.size else float("nan"),
        "n_rolling": int(finite.size),
        "disclaimer": "Rank IC ≠ alpha. Historical Sharpe alone cannot approve.",
    }
