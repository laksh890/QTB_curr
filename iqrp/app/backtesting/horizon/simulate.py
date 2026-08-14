"""Vectorized position simulation for horizon research sweeps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.performance.trade_metrics import trades_from_positions


SignalFn = Callable[[pd.DataFrame, Mapping[str, Any]], np.ndarray]


def default_momentum_signal(frame: pd.DataFrame, params: Mapping[str, Any]) -> np.ndarray:
    """Signed momentum on close; holding applied by ``apply_holding``."""
    lookback = int(params.get("lookback", 1))
    close = frame["close"].to_numpy(dtype=np.float64)
    sig = np.zeros(close.size, dtype=np.float64)
    for i in range(lookback, close.size):
        past = close[i - lookback]
        if past > 0:
            sig[i] = float(np.sign(close[i] / past - 1.0))
    return sig


def apply_holding(signal: np.ndarray, holding_bars: int) -> np.ndarray:
    """Hold each non-zero signal for ``holding_bars`` bars (allow reverse/flat)."""
    h = max(int(holding_bars), 1)
    pos = np.zeros_like(signal, dtype=np.float64)
    i = 0
    n = signal.size
    while i < n:
        s = float(signal[i])
        if abs(s) < 1e-15:
            i += 1
            continue
        end = min(i + h, n)
        pos[i:end] = s
        i = end
    return pos


def simulate_positions(
    frame: pd.DataFrame,
    *,
    signal_fn: SignalFn | None = None,
    params: Mapping[str, Any] | None = None,
    holding_bars: int = 1,
    allow_short: bool = True,
) -> dict[str, Any]:
    """Build positions, gross returns, trades from a single-instrument frame."""
    params = dict(params or {})
    fn = signal_fn or default_momentum_signal
    g = frame.sort_values("timestamp").reset_index(drop=True)
    close = g["close"].to_numpy(dtype=np.float64)
    rets = np.zeros_like(close)
    rets[1:] = close[1:] / close[:-1] - 1.0
    raw = fn(g, params)
    if not allow_short:
        raw = np.maximum(raw, 0.0)
    pos = apply_holding(raw, holding_bars)
    # strategy return: previous position * current return (no lookahead)
    strat = np.zeros_like(rets)
    strat[1:] = pos[:-1] * rets[1:]
    turnover = np.abs(np.diff(pos, prepend=0.0))
    trades = trades_from_positions(pos, rets)
    # enrich sides / times
    ts = pd.to_datetime(g["timestamp"], utc=True)
    enriched = []
    for t in trades:
        start = int(t["start"])
        end = int(t["end"])
        side = "LONG" if t["direction"] > 0 else "SHORT"
        entry = ts.iloc[start]
        exit_ = ts.iloc[min(end, len(ts) - 1)]
        enriched.append(
            {
                **t,
                "side": side,
                "entry_time": entry,
                "exit_time": exit_,
                "holding_seconds": float((exit_ - entry).total_seconds()),
                "holding": float(t["holding"]),
                "pnl": float(t["pnl"]) if np.isfinite(t["pnl"]) else 0.0,
            }
        )
    return {
        "frame": g,
        "signal": raw,
        "positions": pos,
        "returns": rets,
        "gross_returns": strat,
        "turnover_per_period": turnover,
        "trades": enriched,
        "timestamps": ts,
    }


__all__ = [
    "SignalFn",
    "apply_holding",
    "default_momentum_signal",
    "simulate_positions",
]
