"""Smart order router: venue selection, scoring, allocation, and fallback.

ROUTING SAFETY — never route if:
- venue unavailable / halted / trading disabled
- instrument unavailable
- order type unsupported
- invalid price / quantity
- kill switch engaged
- risk reject callback returns False

Execution never generates alpha and never overrides hard risk limits.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from iqrp.app.execution.smart_routing.allocation import (
    AllocationMode,
    AllocationPlan,
    VenueAllocation,
    allocate_quantity,
)
from iqrp.app.execution.smart_routing.cost_model import VenueCostEstimate, estimate_venue_cost
from iqrp.app.execution.smart_routing.fallback import FallbackChain, build_fallback_chain
from iqrp.app.execution.smart_routing.liquidity import LiquiditySnapshot, assess_liquidity
from iqrp.app.execution.smart_routing.scoring import (
    ScoreWeights,
    VenueScore,
    rank_venues,
    score_venue,
)
from iqrp.app.execution.smart_routing.venue import Venue, VenueInterface, as_venue
from iqrp.app.execution.types import KillSwitch, OrderType, Side, Urgency

RiskCheck = Callable[[Any, Venue], bool | tuple[bool, str]]


@dataclass(slots=True)
class RoutingOrder:
    """Minimal order view accepted by SmartRouter.

    Duck-typed orders (e.g. order_manager.Order) work if they expose the
    same attributes: instrument, side, quantity, order_type, price.
    """

    instrument: str
    side: Side | str
    quantity: float
    order_type: OrderType | str = OrderType.MARKET
    price: float | None = None
    order_id: str = ""
    strategy_id: str = ""
    portfolio_id: str = ""
    urgency: Urgency | str = Urgency.NORMAL
    account_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RejectionReason:
    """Why a venue or the whole order was rejected for routing."""

    code: str
    message: str
    venue_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "venue_id": self.venue_id,
        }


@dataclass(slots=True)
class RoutingDecision:
    """Result of ``SmartRouter.route``."""

    accepted: bool
    order_id: str
    instrument: str
    side: str
    quantity: float
    mode: AllocationMode
    primary_venue_id: str | None
    allocations: list[VenueAllocation] = field(default_factory=list)
    scores: list[VenueScore] = field(default_factory=list)
    costs: dict[str, VenueCostEstimate] = field(default_factory=dict)
    liquidity: dict[str, LiquiditySnapshot] = field(default_factory=dict)
    fallback: FallbackChain | None = None
    rejections: list[RejectionReason] = field(default_factory=list)
    residual_qty: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def plan(self) -> AllocationPlan:
        return AllocationPlan(
            mode=self.mode,
            allocations=list(self.allocations),
            residual_qty=self.residual_qty,
            total_allocated=float(sum(a.quantity for a in self.allocations)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "order_id": self.order_id,
            "instrument": self.instrument,
            "side": self.side,
            "quantity": float(self.quantity),
            "mode": self.mode,
            "primary_venue_id": self.primary_venue_id,
            "allocations": [a.to_dict() for a in self.allocations],
            "scores": [s.to_dict() for s in self.scores],
            "costs": {k: v.to_dict() for k, v in self.costs.items()},
            "liquidity": {k: v.to_dict() for k, v in self.liquidity.items()},
            "fallback": self.fallback.to_dict() if self.fallback else None,
            "rejections": [r.to_dict() for r in self.rejections],
            "residual_qty": float(self.residual_qty),
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }


def _order_attr(order: Any, name: str, default: Any = None) -> Any:
    if isinstance(order, Mapping):
        return order.get(name, default)
    return getattr(order, name, default)


def normalize_order(order: Any) -> RoutingOrder:
    """Coerce an order-like object into RoutingOrder."""
    if isinstance(order, RoutingOrder):
        return order
    return RoutingOrder(
        instrument=str(_order_attr(order, "instrument", "")),
        side=_order_attr(order, "side", Side.BUY),
        quantity=float(_order_attr(order, "quantity", 0.0) or 0.0),
        order_type=_order_attr(order, "order_type", OrderType.MARKET),
        price=_order_attr(order, "price", None),
        order_id=str(_order_attr(order, "order_id", "") or ""),
        strategy_id=str(_order_attr(order, "strategy_id", "") or ""),
        portfolio_id=str(_order_attr(order, "portfolio_id", "") or ""),
        urgency=_order_attr(order, "urgency", Urgency.NORMAL),
        account_id=str(_order_attr(order, "account_id", "") or ""),
        metadata=dict(_order_attr(order, "metadata", {}) or {}),
    )


def _price_on_tick(price: float, tick: float) -> bool:
    if tick <= 0:
        return True
    # Allow floating error around tick multiples
    n = round(price / tick)
    return abs(price - n * tick) <= max(tick * 1e-6, 1e-10)


def _qty_on_lot(qty: float, lot: float) -> bool:
    if lot <= 0:
        return True
    n = round(qty / lot)
    return abs(qty - n * lot) <= max(lot * 1e-6, 1e-10)


class SmartRouter:
    """Institutional smart order router.

    Parameters
    ----------
    weights:
        Scoring weights for venue ranking.
    mode:
        ``single`` or ``multi`` venue allocation.
    impact_coeff:
        Square-root impact coefficient.
    global_kill_switch:
        When True, all routing is rejected.
    risk_check:
        Optional callback ``(order, venue) -> bool | (bool, reason)``.
        False / reject reasons block that venue (hard risk — never overridden).
    max_fallbacks:
        Maximum venues in the fallback chain.
    max_venues:
        Cap on venues used in a multi-venue split.
    """

    def __init__(
        self,
        *,
        weights: ScoreWeights | Mapping[str, float] | None = None,
        mode: AllocationMode = "single",
        impact_coeff: float = 0.1,
        global_kill_switch: bool = False,
        kill_switch: KillSwitch | None = None,
        risk_check: RiskCheck | None = None,
        max_fallbacks: int = 5,
        max_venues: int | None = None,
        allow_partial_route: bool = True,
    ) -> None:
        self.weights = (
            weights if isinstance(weights, ScoreWeights) else ScoreWeights.from_mapping(weights)
        )
        self.mode: AllocationMode = mode
        self.impact_coeff = float(impact_coeff)
        self.kill_switch = kill_switch or KillSwitch()
        if global_kill_switch:
            self.kill_switch.engage_global("global kill switch engaged")
        self.risk_check = risk_check
        self.max_fallbacks = int(max_fallbacks)
        self.max_venues = max_venues
        self.allow_partial_route = bool(allow_partial_route)

    @property
    def global_kill_switch(self) -> bool:
        return bool(self.kill_switch.global_halt)

    @global_kill_switch.setter
    def global_kill_switch(self, engaged: bool) -> None:
        if engaged:
            self.kill_switch.engage_global("global kill switch engaged")
        else:
            self.kill_switch.clear_global()

    def set_kill_switch(self, engaged: bool) -> None:
        """Engage or clear the router-level global kill switch."""
        self.global_kill_switch = bool(engaged)

    def route(
        self,
        order: Any,
        venues: Sequence[Venue | VenueInterface | Mapping[str, Any]],
        *,
        mode: AllocationMode | None = None,
    ) -> RoutingDecision:
        """Route ``order`` across ``venues`` and return a RoutingDecision.

        Never routes when safety checks fail. Urgency may bias weights toward
        fill probability / latency but never bypasses hard rejects.
        """
        ro = normalize_order(order)
        alloc_mode: AllocationMode = mode or self.mode
        ts = datetime.now(UTC)
        rejections: list[RejectionReason] = []

        # --- Global / order-level safety ---
        blocked, ks_reason = self.kill_switch.is_blocked(
            account_id=ro.account_id or None,
            strategy_id=ro.strategy_id or None,
        )
        if blocked:
            return self._reject(
                ro,
                alloc_mode,
                [RejectionReason("kill_switch", ks_reason or "kill switch engaged")],
                ts,
            )

        order_rejects = self._validate_order(ro)
        if order_rejects:
            return self._reject(ro, alloc_mode, order_rejects, ts)

        side = Side.parse(ro.side)
        order_type = OrderType.parse(ro.order_type)
        urgency = (
            ro.urgency
            if isinstance(ro.urgency, Urgency)
            else (
                Urgency(str(ro.urgency).upper())
                if str(ro.urgency).upper() in Urgency.__members__
                else Urgency.NORMAL
            )
        )
        weights = self._urgency_adjusted_weights(urgency)

        normalized_venues = [as_venue(v) for v in venues]
        if not normalized_venues:
            return self._reject(
                ro,
                alloc_mode,
                [RejectionReason("no_venues", "no venues provided")],
                ts,
            )

        eligible: list[Venue] = []
        for venue in normalized_venues:
            ok, reason = self._venue_eligible(ro, venue, side, order_type)
            if not ok:
                rejections.append(
                    RejectionReason(
                        reason or "venue_ineligible",
                        reason or "venue ineligible",
                        venue_id=venue.venue_id,
                    )
                )
                continue
            eligible.append(venue)

        if not eligible:
            return self._reject(ro, alloc_mode, rejections, ts)

        # --- Cost / liquidity / score ---
        costs: dict[str, VenueCostEstimate] = {}
        liquidity: dict[str, LiquiditySnapshot] = {}
        scores: list[VenueScore] = []
        peer_prices: list[float] = []

        for venue in eligible:
            cost = estimate_venue_cost(
                venue,
                side=side,
                quantity=float(ro.quantity),
                order_type=order_type,
                price=ro.price,
                impact_coeff=self.impact_coeff,
            )
            costs[venue.venue_id] = cost
            if cost.expected_price > 0:
                peer_prices.append(cost.expected_price)

            liq = assess_liquidity(
                venue,
                instrument=ro.instrument,
                quantity=float(ro.quantity),
                side=side,
                max_participation=venue.max_participation,
            )
            liquidity[venue.venue_id] = liq

        for venue in eligible:
            cost = costs[venue.venue_id]
            liq = liquidity[venue.venue_id]
            if liq.fillable_qty <= 0:
                rejections.append(
                    RejectionReason(
                        "no_liquidity",
                        "venue has no fillable liquidity",
                        venue_id=venue.venue_id,
                    )
                )
                continue
            scores.append(
                score_venue(
                    venue,
                    cost=cost,
                    liquidity=liq,
                    weights=weights,
                    is_buy=(side == Side.BUY),
                    peer_prices=peer_prices,
                )
            )

        if not scores:
            return self._reject(ro, alloc_mode, rejections, ts)

        scores = rank_venues(scores)
        lot_sizes = {v.venue_id: float(v.venue_state.lot_size) for v in eligible}
        min_qtys = {v.venue_id: float(v.venue_state.min_qty) for v in eligible}

        plan = allocate_quantity(
            float(ro.quantity),
            scores,
            liquidity,
            mode=alloc_mode,
            lot_sizes=lot_sizes,
            min_qty=min_qtys,
            max_venues=self.max_venues,
        )

        if not plan.allocations:
            rejections.append(RejectionReason("allocation_empty", "no quantity could be allocated"))
            return self._reject(
                ro, alloc_mode, rejections, ts, scores=scores, costs=costs, liquidity=liquidity
            )

        if plan.residual_qty > 0 and not self.allow_partial_route:
            rejections.append(
                RejectionReason(
                    "insufficient_aggregate_liquidity",
                    f"residual {plan.residual_qty} not allowed under allow_partial_route=False",
                )
            )
            return self._reject(
                ro, alloc_mode, rejections, ts, scores=scores, costs=costs, liquidity=liquidity
            )

        primary = plan.allocations[0].venue_id
        fallback = build_fallback_chain(
            scores,
            primary_venue_id=primary,
            max_fallbacks=self.max_fallbacks,
        )

        return RoutingDecision(
            accepted=True,
            order_id=ro.order_id,
            instrument=ro.instrument,
            side=side.value,
            quantity=float(ro.quantity),
            mode=alloc_mode,
            primary_venue_id=primary,
            allocations=list(plan.allocations),
            scores=scores,
            costs=costs,
            liquidity=liquidity,
            fallback=fallback,
            rejections=rejections,
            residual_qty=float(plan.residual_qty),
            timestamp=ts,
            metadata={
                "urgency": urgency.value,
                "order_type": order_type.value,
                "impact_coeff": self.impact_coeff,
            },
        )

    def route_residual(
        self,
        order: Any,
        venues: Sequence[Venue | VenueInterface | Mapping[str, Any]],
        *,
        exclude_venues: Iterable[str] | None = None,
        residual_qty: float | None = None,
        mode: AllocationMode | None = None,
    ) -> RoutingDecision:
        """Re-route residual quantity after partial fills or venue failure."""
        ro = normalize_order(order)
        qty = float(residual_qty) if residual_qty is not None else float(ro.quantity)
        excluded = {str(x) for x in (exclude_venues or [])}
        filtered = []
        for v in venues:
            venue = as_venue(v)
            if venue.venue_id in excluded:
                continue
            filtered.append(venue)
        residual_order = RoutingOrder(
            instrument=ro.instrument,
            side=ro.side,
            quantity=qty,
            order_type=ro.order_type,
            price=ro.price,
            order_id=ro.order_id,
            strategy_id=ro.strategy_id,
            portfolio_id=ro.portfolio_id,
            urgency=ro.urgency,
            account_id=ro.account_id,
            metadata={**ro.metadata, "residual": True},
        )
        return self.route(residual_order, filtered, mode=mode)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _urgency_adjusted_weights(self, urgency: Urgency) -> ScoreWeights:
        """Bias weights toward fill/latency for higher urgency (never bypasses risk)."""
        w = self.weights.normalized()
        if urgency == Urgency.LOW:
            w["price"] += 0.05
            w["fees"] += 0.03
            w["impact"] += 0.02
            w["fill_prob"] = max(w["fill_prob"] - 0.05, 0.01)
            w["latency"] = max(w["latency"] - 0.02, 0.01)
        elif urgency == Urgency.HIGH:
            w["fill_prob"] += 0.08
            w["latency"] += 0.04
            w["price"] = max(w["price"] - 0.06, 0.01)
            w["fees"] = max(w["fees"] - 0.03, 0.01)
        elif urgency == Urgency.CRITICAL:
            w["fill_prob"] += 0.15
            w["latency"] += 0.08
            w["reliability"] += 0.05
            w["price"] = max(w["price"] - 0.12, 0.01)
            w["fees"] = max(w["fees"] - 0.05, 0.01)
            w["impact"] = max(w["impact"] - 0.04, 0.01)
        return ScoreWeights.from_mapping(w)

    def _validate_order(self, ro: RoutingOrder) -> list[RejectionReason]:
        rejects: list[RejectionReason] = []
        if not ro.instrument:
            rejects.append(RejectionReason("invalid_instrument", "instrument is required"))
        try:
            Side.parse(ro.side)
        except ValueError as exc:
            rejects.append(RejectionReason("invalid_side", str(exc)))
        try:
            OrderType.parse(ro.order_type)
        except ValueError as exc:
            rejects.append(RejectionReason("invalid_order_type", str(exc)))

        qty = float(ro.quantity)
        if not (qty > 0) or qty != qty:  # NaN check
            rejects.append(RejectionReason("invalid_quantity", f"quantity must be > 0, got {qty}"))

        ot = None
        try:
            ot = OrderType.parse(ro.order_type)
        except ValueError:
            ot = None
        if ot in {
            OrderType.LIMIT,
            OrderType.STOP,
            OrderType.STOP_LIMIT,
            OrderType.POST_ONLY,
        }:
            if ro.price is None or not (float(ro.price) > 0) or float(ro.price) != float(ro.price):
                rejects.append(
                    RejectionReason("invalid_price", "price must be > 0 for priced order types")
                )
        elif ro.price is not None:
            if not (float(ro.price) > 0) or float(ro.price) != float(ro.price):
                rejects.append(RejectionReason("invalid_price", f"invalid price {ro.price}"))
        return rejects

    def _venue_eligible(
        self,
        ro: RoutingOrder,
        venue: Venue,
        side: Side,
        order_type: OrderType,
    ) -> tuple[bool, str]:
        del side  # reserved for future asymmetric checks
        state = venue.venue_state

        if state.kill_switch:
            return False, "venue_kill_switch"
        blocked, reason = self.kill_switch.is_blocked(venue=venue.venue_id)
        if blocked:
            return False, reason or "venue_kill_switch"
        if not state.available:
            return False, "venue_unavailable"
        if state.halted:
            return False, "venue_halted"
        if not state.trading_enabled:
            return False, "trading_disabled"
        if not state.supports_instrument(ro.instrument):
            return False, "instrument_unavailable"
        if not state.supports_order_type(order_type):
            return False, "order_type_unsupported"

        qty = float(ro.quantity)
        if qty < float(state.min_qty):
            return False, "qty_below_min"
        if qty > float(state.max_qty):
            return False, "qty_above_max"
        if not _qty_on_lot(qty, float(state.lot_size)):
            # Multi-venue allocation will lot-round; for single full-qty check be strict
            # when mode is single — still allow multi path. Soft-reject only if completely
            # incompatible (lot larger than qty).
            if float(state.lot_size) > qty:
                return False, "invalid_quantity_lot"

        if ro.price is not None:
            px = float(ro.price)
            if px <= 0 or px != px:
                return False, "invalid_price"
            if not _price_on_tick(px, float(state.tick_size)):
                return False, "invalid_price_tick"

        if self.risk_check is not None:
            result = self.risk_check(ro, venue)
            if isinstance(result, tuple):
                ok, reason = result
                if not ok:
                    return False, reason or "risk_reject"
            elif not result:
                return False, "risk_reject"

        return True, ""

    def _reject(
        self,
        ro: RoutingOrder,
        mode: AllocationMode,
        rejections: list[RejectionReason],
        ts: datetime,
        *,
        scores: list[VenueScore] | None = None,
        costs: dict[str, VenueCostEstimate] | None = None,
        liquidity: dict[str, LiquiditySnapshot] | None = None,
    ) -> RoutingDecision:
        side_val = str(getattr(ro.side, "value", ro.side))
        return RoutingDecision(
            accepted=False,
            order_id=ro.order_id,
            instrument=ro.instrument,
            side=side_val,
            quantity=float(ro.quantity),
            mode=mode,
            primary_venue_id=None,
            allocations=[],
            scores=list(scores or []),
            costs=dict(costs or {}),
            liquidity=dict(liquidity or {}),
            fallback=None,
            rejections=list(rejections),
            residual_qty=float(ro.quantity),
            timestamp=ts,
            metadata={"rejected": True},
        )


__all__ = [
    "RejectionReason",
    "RiskCheck",
    "RoutingDecision",
    "RoutingOrder",
    "SmartRouter",
    "normalize_order",
]
