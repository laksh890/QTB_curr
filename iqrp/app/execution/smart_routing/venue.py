"""Venue abstraction and SimulatedVenue for smart order routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from iqrp.app.execution.smart_routing.venue_state import VenueState
from iqrp.app.execution.types import OrderType, Side


class VenueResponseStatus(str, Enum):
    ACK = "ACK"
    FILL = "FILL"
    PARTIAL_FILL = "PARTIAL_FILL"
    REJECT = "REJECT"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class VenueOrderRequest:
    """Minimal order payload sent to a venue."""

    instrument: str
    side: Side
    quantity: float
    order_type: OrderType
    price: float | None = None
    client_order_id: str = ""
    order_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VenueResponse:
    """Venue acknowledgment / fill / reject response."""

    status: VenueResponseStatus
    venue_id: str
    venue_order_id: str
    client_order_id: str = ""
    filled_qty: float = 0.0
    fill_price: float | None = None
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Venue:
    """Static venue configuration plus mutable runtime state."""

    venue_id: str
    name: str = ""
    state: VenueState | None = None
    preference: float = 1.0  # relative preference weight
    max_participation: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.venue_id
        if self.state is None:
            self.state = VenueState(venue_id=self.venue_id)

    @property
    def venue_state(self) -> VenueState:
        assert self.state is not None
        return self.state


@runtime_checkable
class VenueInterface(Protocol):
    """Common interface every execution venue must implement."""

    @property
    def venue_id(self) -> str: ...

    def get_state(self) -> VenueState: ...

    def supports_instrument(self, instrument: str) -> bool: ...

    def supports_order_type(self, order_type: OrderType | str) -> bool: ...

    def submit(self, request: VenueOrderRequest) -> VenueResponse: ...

    def cancel(self, venue_order_id: str) -> VenueResponse: ...


@dataclass
class SimulatedVenue:
    """Testing venue that can ACK, FILL, or REJECT orders.

    Configure ``mode``:
    - ``ack``: acknowledge without fill
    - ``fill``: fully fill at ask (buy) / bid (sell) or limit price
    - ``partial``: fill ``partial_fraction`` of quantity
    - ``reject``: reject every order

    Convenience: pass ``instruments`` (and/or omit ``state``) for quick setup::

        SimulatedVenue(venue_id='SIM', instruments={'AAPL'}, mode='fill')
    """

    venue_id: str
    state: VenueState | None = None
    mode: str = "fill"
    partial_fraction: float = 0.5
    reject_reason: str = "simulated_reject"
    instruments: set[str] | frozenset[str] | list[str] | None = None
    mid: float | None = None
    spread: float | None = None
    available_qty: float = 1e12
    adv: float = 1e6
    _orders: dict[str, VenueOrderRequest] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        inst = {str(i).upper() for i in (self.instruments or [])}
        if self.state is None:
            mid = float(self.mid) if self.mid is not None else 100.0
            half = 0.5 * float(self.spread if self.spread is not None else 0.02)
            self.state = VenueState(
                venue_id=self.venue_id,
                instruments=inst,
                available_qty=float(self.available_qty),
                adv=float(self.adv),
                mid=mid,
                bid=mid - half,
                ask=mid + half,
            )
        elif inst:
            self.state.instruments |= inst
        if self.mid is not None and self.state.mid is None:
            self.state.mid = float(self.mid)
        if self.spread is not None and self.state.bid is None and self.state.mid is not None:
            half = 0.5 * float(self.spread)
            self.state.bid = float(self.state.mid) - half
            self.state.ask = float(self.state.mid) + half

    def get_state(self) -> VenueState:
        assert self.state is not None
        return self.state

    def supports_instrument(self, instrument: str) -> bool:
        return self.state.supports_instrument(instrument)

    def supports_order_type(self, order_type: OrderType | str) -> bool:
        return self.state.supports_order_type(order_type)

    def submit(self, request: VenueOrderRequest) -> VenueResponse:
        if not self.state.is_routable():
            return VenueResponse(
                status=VenueResponseStatus.REJECT,
                venue_id=self.venue_id,
                venue_order_id="",
                client_order_id=request.client_order_id,
                reason="venue_not_routable",
            )
        if not self.supports_instrument(request.instrument):
            return VenueResponse(
                status=VenueResponseStatus.REJECT,
                venue_id=self.venue_id,
                venue_order_id="",
                client_order_id=request.client_order_id,
                reason="instrument_unavailable",
            )
        if not self.supports_order_type(request.order_type):
            return VenueResponse(
                status=VenueResponseStatus.REJECT,
                venue_id=self.venue_id,
                venue_order_id="",
                client_order_id=request.client_order_id,
                reason="order_type_unsupported",
            )

        venue_order_id = f"{self.venue_id}-{uuid4().hex[:12]}"
        self._orders[venue_order_id] = request
        mode = str(self.mode).lower()

        if mode == "reject":
            return VenueResponse(
                status=VenueResponseStatus.REJECT,
                venue_id=self.venue_id,
                venue_order_id=venue_order_id,
                client_order_id=request.client_order_id,
                reason=self.reject_reason,
            )

        if mode == "ack":
            return VenueResponse(
                status=VenueResponseStatus.ACK,
                venue_id=self.venue_id,
                venue_order_id=venue_order_id,
                client_order_id=request.client_order_id,
            )

        fill_price = self._fill_price(request)
        if mode == "partial":
            frac = min(max(float(self.partial_fraction), 0.0), 1.0)
            qty = float(request.quantity) * frac
            return VenueResponse(
                status=VenueResponseStatus.PARTIAL_FILL,
                venue_id=self.venue_id,
                venue_order_id=venue_order_id,
                client_order_id=request.client_order_id,
                filled_qty=qty,
                fill_price=fill_price,
            )

        # default: fill
        return VenueResponse(
            status=VenueResponseStatus.FILL,
            venue_id=self.venue_id,
            venue_order_id=venue_order_id,
            client_order_id=request.client_order_id,
            filled_qty=float(request.quantity),
            fill_price=fill_price,
        )

    def cancel(self, venue_order_id: str) -> VenueResponse:
        order = self._orders.pop(venue_order_id, None)
        if order is None:
            return VenueResponse(
                status=VenueResponseStatus.REJECT,
                venue_id=self.venue_id,
                venue_order_id=venue_order_id,
                reason="unknown_order",
            )
        return VenueResponse(
            status=VenueResponseStatus.CANCELLED,
            venue_id=self.venue_id,
            venue_order_id=venue_order_id,
            client_order_id=order.client_order_id,
        )

    def _fill_price(self, request: VenueOrderRequest) -> float:
        self.state.ensure_quotes()
        if request.order_type == OrderType.LIMIT and request.price is not None:
            return float(request.price)
        if request.side == Side.BUY:
            if self.state.ask is not None:
                return float(self.state.ask)
        else:
            if self.state.bid is not None:
                return float(self.state.bid)
        if self.state.mid is not None:
            return float(self.state.mid)
        if request.price is not None:
            return float(request.price)
        return 0.0


def as_venue(obj: Venue | VenueInterface | dict[str, Any]) -> Venue:
    """Normalize venue-like inputs into a Venue dataclass."""
    if isinstance(obj, Venue):
        return obj
    if isinstance(obj, dict):
        state_data = dict(obj.get("state") or {})
        venue_id = str(obj.get("venue_id") or state_data.get("venue_id") or "UNKNOWN")
        state = VenueState(
            venue_id=venue_id,
            available=bool(state_data.get("available", obj.get("available", True))),
            halted=bool(state_data.get("halted", obj.get("halted", False))),
            trading_enabled=bool(
                state_data.get("trading_enabled", obj.get("trading_enabled", True))
            ),
            latency_ms=float(state_data.get("latency_ms", obj.get("latency_ms", 5.0))),
            liquidity_score=float(
                state_data.get("liquidity_score", obj.get("liquidity_score", 1.0))
            ),
            reliability=float(state_data.get("reliability", obj.get("reliability", 1.0))),
            fill_probability=float(
                state_data.get("fill_probability", obj.get("fill_probability", 0.95))
            ),
            bid=state_data.get("bid", obj.get("bid")),
            ask=state_data.get("ask", obj.get("ask")),
            mid=state_data.get("mid", obj.get("mid")),
            spread_bps=state_data.get("spread_bps", obj.get("spread_bps")),
            available_qty=float(state_data.get("available_qty", obj.get("available_qty", 0.0))),
            adv=float(state_data.get("adv", obj.get("adv", 0.0))),
            volatility=float(state_data.get("volatility", obj.get("volatility", 0.0))),
            instruments=set(state_data.get("instruments") or obj.get("instruments") or []),
            supported_order_types=set(
                state_data.get("supported_order_types") or obj.get("supported_order_types") or []
            ),
            fee_bps=float(state_data.get("fee_bps", obj.get("fee_bps", 1.0))),
            maker_fee_bps=float(state_data.get("maker_fee_bps", obj.get("maker_fee_bps", 0.5))),
            taker_fee_bps=float(state_data.get("taker_fee_bps", obj.get("taker_fee_bps", 1.0))),
            tick_size=float(state_data.get("tick_size", obj.get("tick_size", 0.01))),
            lot_size=float(state_data.get("lot_size", obj.get("lot_size", 1.0))),
            min_qty=float(state_data.get("min_qty", obj.get("min_qty", 1.0))),
            max_qty=float(state_data.get("max_qty", obj.get("max_qty", 1e12))),
            kill_switch=bool(state_data.get("kill_switch", obj.get("kill_switch", False))),
        )
        return Venue(
            venue_id=venue_id,
            name=str(obj.get("name") or venue_id),
            state=state,
            preference=float(obj.get("preference", 1.0)),
            max_participation=float(obj.get("max_participation", 1.0)),
            metadata=dict(obj.get("metadata") or {}),
        )
    # VenueInterface duck type
    state = obj.get_state()
    return Venue(venue_id=obj.venue_id, name=obj.venue_id, state=state)


__all__ = [
    "SimulatedVenue",
    "Venue",
    "VenueInterface",
    "VenueOrderRequest",
    "VenueResponse",
    "VenueResponseStatus",
    "as_venue",
]
