"""Bid/ask spread generation."""

from __future__ import annotations

import numpy as np


class SpreadModel:
    """Dynamic spread as a function of volatility and liquidity stress."""

    def __init__(
        self,
        *,
        base_spread_bps: float = 5.0,
        min_spread_bps: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.base_spread_bps = base_spread_bps
        self.min_spread_bps = min_spread_bps
        self.rng = rng or np.random.default_rng()

    def spreads_bps(
        self,
        mid: np.ndarray,
        volatility: np.ndarray,
        *,
        liquidity_stress: np.ndarray | None = None,
    ) -> np.ndarray:
        mid_arr = np.asarray(mid, dtype=np.float64)
        vol = np.asarray(volatility, dtype=np.float64)
        if vol.ndim > 1:
            vol = vol.reshape(len(mid_arr), -1)[:, 0]
        if len(vol) == len(mid_arr) - 1:
            vol = np.concatenate([[vol[0]], vol])
        stress = (
            np.ones_like(mid_arr)
            if liquidity_stress is None
            else np.asarray(liquidity_stress, dtype=np.float64)
        )
        # Higher vol / stress widens spreads
        bps = self.base_spread_bps * (1.0 + 2.0 * vol) * stress
        bps = bps * (1.0 + 0.05 * self.rng.standard_normal(len(mid_arr)))
        return np.maximum(bps, self.min_spread_bps)

    def bid_ask(self, mid: np.ndarray, spreads_bps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half = (np.asarray(spreads_bps) / 10_000.0) * np.asarray(mid) / 2.0
        bid = np.asarray(mid) - half
        ask = np.asarray(mid) + half
        return bid, ask
