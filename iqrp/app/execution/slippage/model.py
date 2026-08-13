"""Core slippage model combining spread, impact, volatility, and liquidity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(slots=True)
class SlippageBreakdown:
    """Component-wise slippage in price units and bps."""

    spread: float = 0.0
    temporary_impact: float = 0.0
    permanent_impact: float = 0.0
    volatility: float = 0.0
    liquidity: float = 0.0
    size: float = 0.0
    participation: float = 0.0
    delay: float = 0.0
    total: float = 0.0
    total_bps: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spread": self.spread,
            "temporary_impact": self.temporary_impact,
            "permanent_impact": self.permanent_impact,
            "volatility": self.volatility,
            "liquidity": self.liquidity,
            "size": self.size,
            "participation": self.participation,
            "delay": self.delay,
            "total": self.total,
            "total_bps": self.total_bps,
            "metadata": dict(self.metadata),
        }


class ExecutionSlippageModel:
    """Configurable multi-factor slippage model (price units, adverse to the trade)."""

    def __init__(
        self,
        *,
        impact_coeff: float = 0.1,
        permanent_ratio: float = 0.5,
        vol_coeff: float = 0.25,
        liquidity_coeff: float = 0.15,
        delay_coeff: float = 0.05,
        participation_coeff: float = 0.35,
        half_spread: bool = True,
    ) -> None:
        self.impact_coeff = float(impact_coeff)
        self.permanent_ratio = float(permanent_ratio)
        self.vol_coeff = float(vol_coeff)
        self.liquidity_coeff = float(liquidity_coeff)
        self.delay_coeff = float(delay_coeff)
        self.participation_coeff = float(participation_coeff)
        self.half_spread = bool(half_spread)

    def estimate(
        self,
        *,
        side: str,
        quantity: float,
        mid: float,
        spread: float = 0.0,
        adv: float = 1e6,
        volatility: float = 0.02,
        liquidity: float = 1.0,
        delay_seconds: float = 0.0,
        horizon_seconds: float = 0.0,
        trading_day_seconds: float = 23400.0,
    ) -> SlippageBreakdown:
        qty = abs(float(quantity))
        mid_f = max(float(mid), 1e-12)
        adv_f = max(float(adv), 1e-12)
        vol = max(float(volatility), 0.0)
        spr = max(float(spread), 0.0)
        liq = max(float(liquidity), 1e-6)
        participation = qty / adv_f

        spread_px = (0.5 * spr) if self.half_spread else spr
        temp = self.impact_coeff * vol * mid_f * np.sqrt(participation)
        perm = self.permanent_ratio * temp
        # Volatility path risk over delay / horizon
        delay = max(float(delay_seconds), 0.0)
        horizon = max(float(horizon_seconds), delay)
        day = max(float(trading_day_seconds), 1.0)
        time_frac = np.sqrt(horizon / day) if horizon > 0 else 0.0
        vol_px = self.vol_coeff * vol * mid_f * time_frac
        # Thin liquidity surcharge
        liq_px = self.liquidity_coeff * mid_f * participation / liq
        size_px = 0.5 * temp  # order-size component embedded in sqrt impact
        part_px = self.participation_coeff * vol * mid_f * participation
        delay_px = self.delay_coeff * vol * mid_f * np.sqrt(delay / day) if delay > 0 else 0.0

        total = float(
            spread_px + temp + vol_px + liq_px + size_px + part_px + delay_px
        )
        # permanent tracked separately (informational; included partially via temp path)
        total_with_perm_info = total  # permanent is subset attribution of temp for TCA
        bps = total_with_perm_info / mid_f * 1e4
        return SlippageBreakdown(
            spread=float(spread_px),
            temporary_impact=float(temp),
            permanent_impact=float(perm),
            volatility=float(vol_px),
            liquidity=float(liq_px),
            size=float(size_px),
            participation=float(part_px),
            delay=float(delay_px),
            total=float(total),
            total_bps=float(bps),
            metadata={
                "side": str(side).lower(),
                "quantity": qty,
                "participation_rate": float(participation),
                "mid": mid_f,
            },
        )

    def execution_price(
        self,
        *,
        side: str,
        quantity: float,
        mid: float,
        spread: float = 0.0,
        adv: float = 1e6,
        volatility: float = 0.02,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Side-signed execution price incorporating expected slippage."""
        br = self.estimate(
            side=side,
            quantity=quantity,
            mid=mid,
            spread=spread,
            adv=adv,
            volatility=volatility,
            **kwargs,
        )
        sign = 1.0 if str(side).lower() in {"buy", "b", "long"} else -1.0
        px = float(mid) + sign * br.total
        return {
            "price": px,
            "slippage": float(br.total),
            "slippage_bps": float(br.total_bps),
            "temporary_impact": float(br.temporary_impact),
            "permanent_impact": float(br.permanent_impact),
            "participation": float(br.metadata.get("participation_rate", 0.0)),
        }


# Back-compat alias
SlippageModel = ExecutionSlippageModel


def combine_components(components: Mapping[str, float], *, mid: float) -> dict[str, float]:
    """Sum named slippage components (price units) into total / bps."""
    total = float(sum(max(float(v), 0.0) for v in components.values()))
    mid_f = max(float(mid), 1e-12)
    return {"total": total, "total_bps": total / mid_f * 1e4, "components": dict(components)}


__all__ = [
    "ExecutionSlippageModel",
    "SlippageBreakdown",
    "SlippageModel",
    "combine_components",
]
