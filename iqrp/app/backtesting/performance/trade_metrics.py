"""Trade-level performance metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = [
    "average_holding_period",
    "average_loss",
    "average_win",
    "expectancy",
    "loss_rate",
    "number_of_trades",
    "profit_factor",
    "summarize_trades",
    "trade_frequency",
    "trades_from_positions",
    "turnover",
    "win_rate",
]


def _pnl_array(trades: Any) -> np.ndarray:
    """Extract PnL from a sequence of trades or a numeric array."""
    if trades is None:
        return np.zeros(0, dtype=np.float64)
    if isinstance(trades, Mapping):
        if "pnl" in trades:
            return np.asarray(trades["pnl"], dtype=np.float64).reshape(-1)
        raise TypeError("trade mapping must contain 'pnl'")
    arr = np.asarray(trades, dtype=object)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    first = arr.flat[0]
    if isinstance(first, Mapping):
        return np.asarray([float(t.get("pnl", 0.0)) for t in arr.flat], dtype=np.float64)
    if hasattr(first, "pnl"):
        return np.asarray([float(t.pnl) for t in arr.flat], dtype=np.float64)
    return np.asarray(trades, dtype=np.float64).reshape(-1)


def _holding_array(trades: Any) -> np.ndarray:
    if trades is None:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(trades, dtype=object)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    first = arr.flat[0]
    if isinstance(first, Mapping) and "holding" in first:
        return np.asarray([float(t.get("holding", np.nan)) for t in arr.flat], dtype=np.float64)
    if hasattr(first, "holding"):
        return np.asarray([float(t.holding) for t in arr.flat], dtype=np.float64)
    return np.zeros(0, dtype=np.float64)


def trades_from_positions(
    positions: Any,
    returns: Any | None = None,
) -> list[dict[str, float]]:
    """Infer round-trip trades from a 1-D signed position series.

    A trade starts when position leaves zero and ends when it returns to zero
    or flips sign. Optional ``returns`` attribute PnL as ``sum(pos * ret)``
    over the holding window.
    """
    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    ret = None if returns is None else np.asarray(returns, dtype=np.float64).reshape(-1)
    trades: list[dict[str, float]] = []
    if pos.size == 0:
        return trades

    i = 0
    n = pos.size
    while i < n:
        if abs(pos[i]) < 1e-15:
            i += 1
            continue
        start = i
        sign = np.sign(pos[i])
        i += 1
        while i < n and np.sign(pos[i]) == sign and abs(pos[i]) > 1e-15:
            i += 1
        end = i  # exclusive
        holding = float(end - start)
        if ret is not None:
            sl = slice(start, min(end, ret.size))
            pnl = float(np.sum(pos[sl] * ret[sl]))
        else:
            pnl = float(np.nan)
        trades.append(
            {
                "start": float(start),
                "end": float(end - 1),
                "holding": holding,
                "pnl": pnl,
                "direction": float(sign),
            }
        )
    return trades


def number_of_trades(trades: Any) -> int:
    """Count of trades."""
    return int(_pnl_array(trades).size)


def win_rate(trades: Any) -> float:
    """Fraction of trades with positive PnL."""
    pnl = _pnl_array(trades)
    if pnl.size == 0:
        return 0.0
    return float(np.mean(pnl > 0.0))


def loss_rate(trades: Any) -> float:
    """Fraction of trades with negative PnL."""
    pnl = _pnl_array(trades)
    if pnl.size == 0:
        return 0.0
    return float(np.mean(pnl < 0.0))


def profit_factor(trades: Any) -> float:
    """Gross profits / gross losses."""
    pnl = _pnl_array(trades)
    if pnl.size == 0:
        return 0.0
    gains = float(np.sum(pnl[pnl > 0.0]))
    losses = float(-np.sum(pnl[pnl < 0.0]))
    if losses < 1e-15:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def average_win(trades: Any) -> float:
    """Mean PnL of winning trades."""
    pnl = _pnl_array(trades)
    wins = pnl[pnl > 0.0]
    return float(np.mean(wins)) if wins.size else 0.0


def average_loss(trades: Any) -> float:
    """Mean PnL of losing trades (negative number)."""
    pnl = _pnl_array(trades)
    losses = pnl[pnl < 0.0]
    return float(np.mean(losses)) if losses.size else 0.0


def expectancy(trades: Any) -> float:
    """Win-rate-weighted expectancy per trade."""
    pnl = _pnl_array(trades)
    if pnl.size == 0:
        return 0.0
    return float(np.mean(pnl))


def average_holding_period(trades: Any) -> float:
    """Mean holding period in bars."""
    h = _holding_array(trades)
    if h.size == 0:
        return 0.0
    finite = h[np.isfinite(h)]
    return float(np.mean(finite)) if finite.size else 0.0


def turnover(
    positions: Any,
    *,
    periods_per_year: float | None = None,
) -> float:
    """Mean absolute position change; optionally annualized."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim == 1:
        if pos.size < 2:
            return 0.0
        to = float(np.mean(np.abs(np.diff(pos))))
    elif pos.ndim == 2:
        if pos.shape[0] < 2:
            return 0.0
        to = float(np.mean(np.sum(np.abs(np.diff(pos, axis=0)), axis=1)))
    else:
        raise ValueError("positions must be 1-D or 2-D")
    if periods_per_year is not None:
        to *= float(periods_per_year)
    return to


def trade_frequency(
    trades: Any,
    *,
    n_periods: int,
    periods_per_year: float = 252.0,
) -> float:
    """Annualized trade count."""
    n = number_of_trades(trades)
    if n_periods <= 0:
        return 0.0
    return float(n) * float(periods_per_year) / float(n_periods)


def summarize_trades(
    trades: Any,
    *,
    positions: Any | None = None,
    n_periods: int | None = None,
    periods_per_year: float = 252.0,
) -> dict[str, float]:
    """Trade metrics summary."""
    pnl = _pnl_array(trades)
    n_per = int(n_periods) if n_periods is not None else max(int(pnl.size), 1)
    out: dict[str, float] = {
        "n_trades": float(number_of_trades(trades)),
        "win_rate": win_rate(trades),
        "loss_rate": loss_rate(trades),
        "profit_factor": profit_factor(trades),
        "average_win": average_win(trades),
        "average_loss": average_loss(trades),
        "expectancy": expectancy(trades),
        "average_holding_period": average_holding_period(trades),
        "trade_frequency": trade_frequency(
            trades, n_periods=n_per, periods_per_year=periods_per_year
        ),
    }
    if positions is not None:
        out["turnover"] = turnover(positions)
    return out
