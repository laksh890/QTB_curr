"""Bid-ask spread transaction cost."""

from __future__ import annotations

from typing import Any


def spread_cost(
    *,
    quantity: float,
    mid: float,
    spread: float,
    half_spread: bool = True,
    side: str | None = None,
) -> dict[str, Any]:
    """Spread cost in currency (quantity * spread_px)."""
    qty = abs(float(quantity))
    mid_f = max(float(mid), 1e-12)
    spr = max(float(spread), 0.0)
    px = 0.5 * spr if half_spread else spr
    total = qty * px
    _ = side
    return {
        "name": "spread_cost",
        "total": float(total),
        "spread_px": float(px),
        "spread_bps": float(px / mid_f * 1e4),
        "notional": float(qty * mid_f),
        "half_spread": bool(half_spread),
    }


__all__ = ["spread_cost"]
