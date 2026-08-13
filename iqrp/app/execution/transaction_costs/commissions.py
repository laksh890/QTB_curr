"""Commission cost models for execution TCA."""

from __future__ import annotations

from typing import Any


def commission_cost(
    *,
    quantity: float,
    price: float,
    commission_bps: float = 1.0,
    commission_per_share: float = 0.0,
    min_commission: float = 0.0,
    side: str | None = None,
) -> dict[str, Any]:
    """Commission on traded notional / shares."""
    qty = abs(float(quantity))
    px = max(float(price), 0.0)
    notional = qty * px
    bps = max(float(commission_bps), 0.0)
    per_share = max(float(commission_per_share), 0.0)
    floor = max(float(min_commission), 0.0)
    cost = qty * per_share + notional * (bps / 1e4)
    if floor > 0.0 and notional > 0.0:
        cost = max(cost, floor)
    _ = side
    return {
        "name": "commission_cost",
        "total": float(cost),
        "notional": float(notional),
        "quantity": qty,
        "price": px,
        "commission_bps": bps,
        "commission_per_share": per_share,
        "min_commission": floor,
    }


__all__ = ["commission_cost"]
