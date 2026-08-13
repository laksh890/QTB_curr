"""Liquidity-driven slippage component."""

from __future__ import annotations

from typing import Any


def liquidity_slippage(
    *,
    mid: float,
    quantity: float,
    adv: float,
    liquidity: float = 1.0,
    coeff: float = 0.15,
    depth: float | None = None,
) -> dict[str, Any]:
    """Surcharge when liquidity is thin relative to order size / ADV."""
    mid_f = max(float(mid), 1e-12)
    qty = abs(float(quantity))
    adv_f = max(float(adv), 1e-12)
    liq = max(float(liquidity), 1e-6)
    participation = qty / adv_f
    if depth is not None and float(depth) > 0:
        depth_part = qty / max(float(depth), 1e-12)
        participation = max(participation, depth_part)
    px = float(coeff) * mid_f * participation / liq
    return {
        "name": "liquidity_slippage",
        "slippage": float(px),
        "slippage_bps": float(px / mid_f * 1e4),
        "participation": float(participation),
        "liquidity": liq,
        "coeff": float(coeff),
    }


__all__ = ["liquidity_slippage"]
