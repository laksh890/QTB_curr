"""Temporary / permanent market-impact slippage model."""

from __future__ import annotations

import numpy as np


class SlippageModel:
    """Square-root market impact with temporary and permanent components."""

    def __init__(self, *, impact: float = 0.1, rng: np.random.Generator | None = None) -> None:
        self.impact = float(impact)
        self.rng = rng or np.random.default_rng()

    def execution_price(
        self,
        mid: float,
        side: str,
        quantity: float,
        *,
        adv: float,
        volatility: float,
        spread: float,
    ) -> dict[str, float]:
        """Return fill price and impact decomposition.

        ``side`` is ``buy`` or ``sell``. ``adv`` is average daily volume proxy.
        """
        participation = abs(quantity) / max(adv, 1e-8)
        temp = self.impact * volatility * mid * np.sqrt(participation)
        perm = 0.5 * temp
        noise = 0.1 * spread * self.rng.standard_normal()
        sign = 1.0 if side.lower() == "buy" else -1.0
        px = mid + sign * (0.5 * spread + temp + noise)
        return {
            "price": float(px),
            "temporary_impact": float(temp),
            "permanent_impact": float(perm),
            "participation": float(participation),
        }

    def path_impact(
        self,
        mids: np.ndarray,
        volumes: np.ndarray,
        trade_sizes: np.ndarray,
        volatility: np.ndarray,
    ) -> np.ndarray:
        """Vectorized temporary impact series (price units)."""
        mids_a = np.asarray(mids, dtype=np.float64)
        vols = np.asarray(volumes, dtype=np.float64)
        sizes = np.asarray(trade_sizes, dtype=np.float64)
        vol = np.asarray(volatility, dtype=np.float64)
        n = min(len(mids_a), len(vols), len(sizes), len(vol))
        part = np.abs(sizes[:n]) / np.maximum(vols[:n], 1e-8)
        return self.impact * vol[:n] * mids_a[:n] * np.sqrt(part)
