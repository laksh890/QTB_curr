"""Per-venue expected execution cost model for smart routing.

Computes expected price, fees, spread cost, and market impact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from iqrp.app.execution.smart_routing.venue import Venue
from iqrp.app.execution.types import OrderType, Side


@dataclass(slots=True)
class VenueCostEstimate:
    """Expected cost components for routing quantity to one venue."""

    venue_id: str
    expected_price: float
    mid: float
    spread_bps: float
    fee_bps: float
    fee_notional: float
    impact_bps: float
    impact_notional: float
    spread_notional: float
    total_cost_bps: float
    total_cost_notional: float
    notional: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "expected_price": float(self.expected_price),
            "mid": float(self.mid),
            "spread_bps": float(self.spread_bps),
            "fee_bps": float(self.fee_bps),
            "fee_notional": float(self.fee_notional),
            "impact_bps": float(self.impact_bps),
            "impact_notional": float(self.impact_notional),
            "spread_notional": float(self.spread_notional),
            "total_cost_bps": float(self.total_cost_bps),
            "total_cost_notional": float(self.total_cost_notional),
            "notional": float(self.notional),
        }


def _reference_mid(venue: Venue, price: float | None) -> float:
    state = venue.venue_state
    state.ensure_quotes()
    if state.mid is not None and float(state.mid) > 0:
        return float(state.mid)
    if state.bid is not None and state.ask is not None:
        return 0.5 * (float(state.bid) + float(state.ask))
    if price is not None and float(price) > 0:
        return float(price)
    return 0.0


def _expected_price(
    venue: Venue,
    *,
    side: Side,
    order_type: OrderType,
    limit_price: float | None,
    mid: float,
    half_spread: float,
    impact_bps: float,
) -> float:
    """Aggressive expected execution price including half-spread and impact."""
    impact_frac = impact_bps / 10_000.0
    if order_type == OrderType.LIMIT and limit_price is not None and float(limit_price) > 0:
        # Limit orders: expect fill at limit, plus residual impact if aggressive
        px = float(limit_price)
        if side == Side.BUY:
            return px * (1.0 + 0.25 * impact_frac)
        return px * (1.0 - 0.25 * impact_frac)

    state = venue.venue_state
    if side == Side.BUY:
        base = float(state.ask) if state.ask is not None else mid + half_spread
        return base * (1.0 + impact_frac)
    base = float(state.bid) if state.bid is not None else mid - half_spread
    return base * (1.0 - impact_frac)


def estimate_venue_cost(
    venue: Venue,
    *,
    side: Side | str,
    quantity: float,
    order_type: OrderType | str = OrderType.MARKET,
    price: float | None = None,
    impact_coeff: float = 0.1,
) -> VenueCostEstimate:
    """Estimate expected price, fees, and impact for routing to ``venue``.

    Impact model (square-root participation):
        impact_bps = 1e4 * impact_coeff * sigma * sqrt(qty / ADV)
    """
    side_e = Side.parse(side)
    ot = OrderType.parse(order_type)
    state = venue.venue_state
    state.ensure_quotes()

    qty = max(float(quantity), 0.0)
    mid = _reference_mid(venue, price)
    spread_bps = float(state.spread_bps) if state.spread_bps is not None else 0.0
    if spread_bps <= 0.0 and state.bid is not None and state.ask is not None and mid > 0:
        spread_bps = 10_000.0 * (float(state.ask) - float(state.bid)) / mid

    half_spread = mid * (spread_bps / 20_000.0) if mid > 0 else 0.0

    adv = max(float(state.adv), 0.0)
    sigma = max(float(state.volatility), 0.0)
    if adv > 0.0 and qty > 0.0:
        participation = qty / adv
        impact_bps = 10_000.0 * float(impact_coeff) * sigma * (participation**0.5)
    else:
        # No ADV: scale impact by inverse liquidity score
        liq = max(float(state.liquidity_score), 1e-6)
        impact_bps = 10_000.0 * float(impact_coeff) * max(sigma, 0.01) * (1.0 / liq) * 0.01

    impact_bps = max(float(impact_bps), 0.0)

    # Prefer taker fees for marketable orders; maker for post-only/limit passive
    if ot in {OrderType.POST_ONLY}:
        fee_bps = float(state.maker_fee_bps)
    elif ot in {OrderType.LIMIT, OrderType.GTC, OrderType.GTT} and price is not None:
        # Conservative: assume half maker / half taker for limit
        fee_bps = 0.5 * (float(state.maker_fee_bps) + float(state.taker_fee_bps))
    else:
        fee_bps = float(state.taker_fee_bps if state.taker_fee_bps else state.fee_bps)

    expected_price = _expected_price(
        venue,
        side=side_e,
        order_type=ot,
        limit_price=price,
        mid=mid,
        half_spread=half_spread,
        impact_bps=impact_bps,
    )
    notional = abs(expected_price * qty)
    fee_notional = notional * fee_bps / 10_000.0
    impact_notional = notional * impact_bps / 10_000.0
    spread_notional = notional * (spread_bps / 2.0) / 10_000.0
    total_cost_notional = fee_notional + impact_notional + spread_notional
    total_cost_bps = (total_cost_notional / notional * 10_000.0) if notional > 0 else 0.0

    return VenueCostEstimate(
        venue_id=venue.venue_id,
        expected_price=float(expected_price),
        mid=float(mid),
        spread_bps=float(spread_bps),
        fee_bps=float(fee_bps),
        fee_notional=float(fee_notional),
        impact_bps=float(impact_bps),
        impact_notional=float(impact_notional),
        spread_notional=float(spread_notional),
        total_cost_bps=float(total_cost_bps),
        total_cost_notional=float(total_cost_notional),
        notional=float(notional),
    )


__all__ = [
    "VenueCostEstimate",
    "estimate_venue_cost",
]
