"""Exchange / venue fee models."""

from __future__ import annotations

from typing import Any


def exchange_fees(
    *,
    quantity: float,
    price: float,
    fee_bps: float = 0.3,
    fee_per_share: float = 0.0,
    maker_bps: float | None = None,
    taker_bps: float | None = None,
    liquidity_role: str = "taker",
    min_fee: float = 0.0,
) -> dict[str, Any]:
    """Exchange fees; optional maker/taker differential."""
    qty = abs(float(quantity))
    px = max(float(price), 0.0)
    notional = qty * px
    role = str(liquidity_role).strip().lower()
    if role == "maker" and maker_bps is not None:
        bps = float(maker_bps)
    elif role == "taker" and taker_bps is not None:
        bps = float(taker_bps)
    else:
        bps = max(float(fee_bps), 0.0)
    per_share = max(float(fee_per_share), 0.0)
    cost = qty * per_share + notional * (bps / 1e4)
    floor = max(float(min_fee), 0.0)
    if floor > 0.0 and notional > 0.0:
        cost = max(cost, floor)
    return {
        "name": "exchange_fees",
        "total": float(cost),
        "notional": float(notional),
        "fee_bps": float(bps),
        "fee_per_share": per_share,
        "liquidity_role": role,
    }


__all__ = ["exchange_fees"]
