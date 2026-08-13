"""Drawdown analytics for backtests."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.returns import as_returns, wealth_index

__all__ = [
    "drawdown_series",
    "max_drawdown",
    "average_drawdown",
    "drawdown_episodes",
    "max_drawdown_duration",
    "average_drawdown_duration",
    "recovery_time",
    "ulcer_index",
    "pain_index",
    "time_underwater",
    "summarize_drawdown",
]


def drawdown_series(returns: Any) -> np.ndarray:
    """Running peak-to-trough drawdown (positive fraction)."""
    r = as_returns(returns)
    if r.size == 0:
        return np.zeros(0, dtype=np.float64)
    path = wealth_index(r)
    peak = np.maximum.accumulate(path)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = 1.0 - path / np.maximum(peak, 1e-12)
    return np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)


def max_drawdown(returns: Any) -> float:
    """Maximum drawdown as a positive fraction."""
    dd = drawdown_series(returns)
    return float(np.max(dd)) if dd.size else 0.0


def average_drawdown(returns: Any) -> float:
    """Mean of the drawdown series."""
    dd = drawdown_series(returns)
    return float(np.mean(dd)) if dd.size else 0.0


def drawdown_episodes(returns: Any) -> list[dict[str, Any]]:
    """Identify contiguous underwater episodes.

    Each episode dict has ``start``, ``trough``, ``end``, ``depth``, ``duration``,
    ``recovery`` (bars from trough to first recovered bar, or None if unrecovered).
    """
    dd = drawdown_series(returns)
    if dd.size == 0:
        return []
    underwater = dd > 1e-12
    episodes: list[dict[str, Any]] = []
    i = 0
    n = dd.size
    while i < n:
        if not underwater[i]:
            i += 1
            continue
        start = i
        while i < n and underwater[i]:
            i += 1
        end = i - 1
        trough = int(start + np.argmax(dd[start : end + 1]))
        depth = float(dd[trough])
        if end < n - 1:
            recovery: int | None = int((end + 1) - trough)
        else:
            recovery = None
        episodes.append(
            {
                "start": int(start),
                "trough": trough,
                "end": int(end),
                "depth": depth,
                "duration": int(end - start + 1),
                "recovery": recovery,
            }
        )
    return episodes


def max_drawdown_duration(returns: Any) -> int:
    """Longest contiguous underwater duration in bars."""
    eps = drawdown_episodes(returns)
    if not eps:
        return 0
    return int(max(e["duration"] for e in eps))


def average_drawdown_duration(returns: Any) -> float:
    """Average underwater episode duration in bars."""
    eps = drawdown_episodes(returns)
    if not eps:
        return 0.0
    return float(np.mean([e["duration"] for e in eps]))


def recovery_time(returns: Any) -> float | None:
    """Mean recovery time (trough → new high) across recovered episodes."""
    eps = drawdown_episodes(returns)
    recovered = [e["recovery"] for e in eps if e["recovery"] is not None]
    if not recovered:
        return None
    return float(np.mean(recovered))


def ulcer_index(returns: Any) -> float:
    """Ulcer index: RMS of drawdowns."""
    dd = drawdown_series(returns)
    if dd.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(dd ** 2)))


def pain_index(returns: Any) -> float:
    """Pain index: mean drawdown (alias of average drawdown)."""
    return average_drawdown(returns)


def time_underwater(returns: Any) -> dict[str, float]:
    """Fraction and count of bars spent underwater."""
    dd = drawdown_series(returns)
    if dd.size == 0:
        return {"bars": 0.0, "fraction": 0.0}
    underwater = dd > 1e-12
    bars = float(np.sum(underwater))
    return {"bars": bars, "fraction": float(bars / dd.size)}


def summarize_drawdown(returns: Any) -> dict[str, Any]:
    """Full drawdown summary."""
    tu = time_underwater(returns)
    return {
        "max_drawdown": max_drawdown(returns),
        "average_drawdown": average_drawdown(returns),
        "max_duration": max_drawdown_duration(returns),
        "average_duration": average_drawdown_duration(returns),
        "recovery_time": recovery_time(returns),
        "ulcer_index": ulcer_index(returns),
        "pain_index": pain_index(returns),
        "time_underwater_bars": tu["bars"],
        "time_underwater_fraction": tu["fraction"],
        "n_episodes": len(drawdown_episodes(returns)),
    }
