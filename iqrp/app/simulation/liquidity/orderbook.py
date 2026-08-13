"""Synthetic limit order book generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


class OrderBookGenerator:
    """Build L2-style depth ladders around mid."""

    def __init__(
        self,
        *,
        depth_levels: int = 5,
        base_depth: float = 10.0,
        tick_size: float = 0.01,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.depth_levels = max(1, int(depth_levels))
        self.base_depth = float(base_depth)
        self.tick_size = float(tick_size)
        self.rng = rng or np.random.default_rng()

    def snapshot(
        self,
        mid: float,
        spread: float,
        *,
        liquidity_mult: float = 1.0,
        timestamp: Any = None,
        symbol: str = "SYNTH",
    ) -> list[dict[str, Any]]:
        half = max(spread / 2.0, self.tick_size)
        bid0 = mid - half
        ask0 = mid + half
        rows: list[dict[str, Any]] = []
        for level in range(self.depth_levels):
            decay = np.exp(-0.35 * level)
            size = self.base_depth * liquidity_mult * decay * (0.8 + 0.4 * self.rng.random())
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": "bid",
                    "level": level,
                    "price": max(self.tick_size, bid0 - level * self.tick_size),
                    "size": float(size),
                }
            )
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": "ask",
                    "level": level,
                    "price": ask0 + level * self.tick_size,
                    "size": float(size * (0.9 + 0.2 * self.rng.random())),
                }
            )
        return rows

    def generate_frame(
        self,
        mids: np.ndarray,
        spreads: np.ndarray,
        timestamps: list[Any],
        *,
        symbol: str = "SYNTH",
        liquidity_stress: np.ndarray | None = None,
        stride: int = 10,
    ) -> pl.DataFrame:
        mids_a = np.asarray(mids, dtype=np.float64)
        spreads_a = np.asarray(spreads, dtype=np.float64)
        stress = (
            np.ones(len(mids_a))
            if liquidity_stress is None
            else np.asarray(liquidity_stress, dtype=np.float64)
        )
        rows: list[dict[str, Any]] = []
        for i in range(0, len(mids_a), max(1, stride)):
            mult = 1.0 / max(float(stress[i]), 1e-3)
            rows.extend(
                self.snapshot(
                    float(mids_a[i]),
                    float(spreads_a[i]),
                    liquidity_mult=mult,
                    timestamp=timestamps[i] if i < len(timestamps) else None,
                    symbol=symbol,
                )
            )
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def generate_trades(
        self,
        mids: np.ndarray,
        volumes: np.ndarray,
        timestamps: list[Any],
        *,
        symbol: str = "SYNTH",
        trades_per_bar: int = 3,
    ) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for i, mid in enumerate(np.asarray(mids, dtype=np.float64)):
            bar_vol = float(volumes[i]) if i < len(volumes) else 1.0
            for j in range(trades_per_bar):
                side = "buy" if self.rng.random() > 0.5 else "sell"
                px = mid * (1.0 + self.rng.normal(0.0, 0.0002))
                qty = max(1e-8, bar_vol / trades_per_bar * (0.5 + self.rng.random()))
                rows.append(
                    {
                        "timestamp": timestamps[i] if i < len(timestamps) else None,
                        "symbol": symbol,
                        "price": float(px),
                        "quantity": float(qty),
                        "side": side,
                        "trade_id": f"{i}-{j}",
                    }
                )
        return pl.DataFrame(rows)
