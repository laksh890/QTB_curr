"""Order dataclass for institutional execution.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Never override hard risk limits.
- Urgency influences aggressiveness but NEVER overrides hard risk.
- Idempotent fills/events; no future information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from iqrp.app.execution.order_manager.order_state import OrderState
from iqrp.app.execution.types import OrderType, Side, TimeInForce, Urgency


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str = "ord") -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


@dataclass
class Order:
    """Institutional execution order.

    ``audit`` is an append-only list of per-order event dicts. Global audit
    is owned by :class:`~iqrp.app.execution.order_manager.audit.AuditLog`.
    """

    instrument: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.LIMIT
    price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    venue: str | None = None
    algo: str | None = None
    urgency: Urgency = Urgency.NORMAL
    strategy_id: str | None = None
    portfolio_id: str | None = None
    parent_id: str | None = None
    client_order_id: str | None = None
    idempotency_key: str | None = None
    account_id: str | None = None
    order_id: str = field(default_factory=lambda: _new_id("ord"))
    venue_order_id: str | None = None
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    state: OrderState = OrderState.CREATED
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    submitted_at: str | None = None
    acknowledged_at: str | None = None
    completed_at: str | None = None
    reject_reason: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    audit: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.instrument = str(self.instrument).strip().upper()
        self.quantity = float(self.quantity)
        self.filled_qty = float(self.filled_qty)
        if isinstance(self.side, str):
            self.side = Side(self.side)
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type)
        if isinstance(self.time_in_force, str):
            self.time_in_force = TimeInForce(self.time_in_force)
        if isinstance(self.urgency, str):
            self.urgency = Urgency(self.urgency)
        if isinstance(self.state, str):
            self.state = OrderState(self.state)
        if self.client_order_id is None:
            self.client_order_id = self.order_id
        if self.idempotency_key is None:
            self.idempotency_key = (
                f"{self.strategy_id or 'na'}|{self.instrument}|{self.side.value}|"
                f"{self.quantity}|{self.order_type.value}|{self.price}|{self.client_order_id}"
            )

    @property
    def residual_qty(self) -> float:
        return max(float(self.quantity) - float(self.filled_qty), 0.0)

    @property
    def is_terminal(self) -> bool:
        from iqrp.app.execution.order_manager.order_state import TERMINAL_STATES

        return self.state in TERMINAL_STATES

    @property
    def notional(self) -> float | None:
        px = self.price if self.price is not None else self.avg_fill_price
        if px is None:
            return None
        return abs(float(self.quantity) * float(px))

    def touch(self) -> None:
        self.updated_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "parent_id": self.parent_id,
            "strategy_id": self.strategy_id,
            "portfolio_id": self.portfolio_id,
            "account_id": self.account_id,
            "instrument": self.instrument,
            "side": self.side.value,
            "quantity": float(self.quantity),
            "filled_qty": float(self.filled_qty),
            "residual_qty": self.residual_qty,
            "price": self.price,
            "stop_price": self.stop_price,
            "order_type": self.order_type.value,
            "time_in_force": self.time_in_force.value,
            "venue": self.venue,
            "algo": self.algo,
            "urgency": self.urgency.value,
            "state": self.state.value,
            "client_order_id": self.client_order_id,
            "idempotency_key": self.idempotency_key,
            "venue_order_id": self.venue_order_id,
            "avg_fill_price": self.avg_fill_price,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "submitted_at": self.submitted_at,
            "acknowledged_at": self.acknowledged_at,
            "completed_at": self.completed_at,
            "reject_reason": self.reject_reason,
            "tags": dict(self.tags),
            "audit": list(self.audit),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Order:
        return cls(
            order_id=str(data.get("order_id") or _new_id("ord")),
            parent_id=data.get("parent_id"),
            strategy_id=data.get("strategy_id"),
            portfolio_id=data.get("portfolio_id"),
            account_id=data.get("account_id"),
            instrument=str(data["instrument"]),
            side=Side(data["side"]),
            quantity=float(data["quantity"]),
            filled_qty=float(data.get("filled_qty", 0.0)),
            price=float(data["price"]) if data.get("price") is not None else None,
            stop_price=float(data["stop_price"]) if data.get("stop_price") is not None else None,
            order_type=OrderType(data.get("order_type", OrderType.LIMIT.value)),
            time_in_force=TimeInForce(data.get("time_in_force", TimeInForce.DAY.value)),
            venue=data.get("venue"),
            algo=data.get("algo"),
            urgency=Urgency(data.get("urgency", Urgency.NORMAL.value)),
            state=OrderState(data.get("state", OrderState.CREATED.value)),
            client_order_id=data.get("client_order_id"),
            idempotency_key=data.get("idempotency_key"),
            venue_order_id=data.get("venue_order_id"),
            avg_fill_price=(
                float(data["avg_fill_price"]) if data.get("avg_fill_price") is not None else None
            ),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            submitted_at=data.get("submitted_at"),
            acknowledged_at=data.get("acknowledged_at"),
            completed_at=data.get("completed_at"),
            reject_reason=data.get("reject_reason"),
            tags=dict(data.get("tags") or {}),
            audit=list(data.get("audit") or []),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class OrderSpec:
    """Lightweight order specification produced by ``target_to_orders``."""

    instrument: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.LIMIT
    price: float | None = None
    strategy_id: str | None = None
    portfolio_id: str | None = None
    urgency: Urgency = Urgency.NORMAL
    time_in_force: TimeInForce = TimeInForce.DAY
    venue: str | None = None
    algo: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    def to_order_kwargs(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "price": self.price,
            "strategy_id": self.strategy_id,
            "portfolio_id": self.portfolio_id,
            "urgency": self.urgency,
            "time_in_force": self.time_in_force,
            "venue": self.venue,
            "algo": self.algo,
            "tags": dict(self.tags),
        }


def target_to_orders(
    current: dict[str, float],
    target: dict[str, float],
    *,
    lot_size: float = 1.0,
    min_qty: float = 1.0,
    prices: dict[str, float] | None = None,
    order_type: OrderType = OrderType.LIMIT,
    strategy_id: str | None = None,
    portfolio_id: str | None = None,
    urgency: Urgency = Urgency.NORMAL,
    time_in_force: TimeInForce = TimeInForce.DAY,
    venue: str | None = None,
    round_lots: bool = True,
) -> list[OrderSpec]:
    """Convert current→target position map into delta order specs.

    Uses only present information (current and target). Does not generate
    alpha. Quantities are lot-rounded when ``round_lots`` is True. Orders
    below ``min_qty`` are dropped.

    Urgency is attached for downstream aggressiveness only — it NEVER
    overrides hard risk limits enforced at validation/submit.
    """
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")
    if min_qty < 0:
        raise ValueError("min_qty must be non-negative")

    instruments = sorted(set(current) | set(target))
    specs: list[OrderSpec] = []
    px_map = dict(prices or {})

    for inst in instruments:
        cur = float(current.get(inst, 0.0))
        tgt = float(target.get(inst, 0.0))
        delta = tgt - cur
        if abs(delta) < 1e-12:
            continue
        qty = abs(delta)
        if round_lots:
            qty = (qty // lot_size) * lot_size
        if qty < min_qty - 1e-12:
            continue
        side = Side.BUY if delta > 0 else Side.SELL
        specs.append(
            OrderSpec(
                instrument=str(inst).upper(),
                side=side,
                quantity=float(qty),
                order_type=order_type,
                price=px_map.get(inst),
                strategy_id=strategy_id,
                portfolio_id=portfolio_id,
                urgency=urgency,
                time_in_force=time_in_force,
                venue=venue,
                tags={"delta": float(delta), "current": cur, "target": tgt},
            )
        )
    return specs
