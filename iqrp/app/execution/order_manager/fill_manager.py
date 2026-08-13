"""Idempotent fill application and residual quantity tracking.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Fills are applied idempotently via event_id.
- Never override hard risk limits.
- No future information (fill uses only the fill event itself).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from iqrp.app.core.exceptions import ExecutionError, ValidationError
from iqrp.app.execution.order_manager.order_lifecycle import apply_fill_state

if TYPE_CHECKING:
    from iqrp.app.execution.order_manager.audit import AuditLog
    from iqrp.app.execution.order_manager.order import Order


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Fill:
    order_id: str
    fill_qty: float
    fill_price: float
    event_id: str
    timestamp: str = field(default_factory=_utc_now)
    venue_exec_id: str | None = None
    liquidity_flag: str | None = None
    fees: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "fill_qty": float(self.fill_qty),
            "fill_price": float(self.fill_price),
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "venue_exec_id": self.venue_exec_id,
            "liquidity_flag": self.liquidity_flag,
            "fees": float(self.fees),
            "metadata": dict(self.metadata),
        }


@dataclass
class FillManager:
    """Apply fills idempotently and maintain residual quantity.

    Duplicate ``event_id`` values are ignored (idempotent). Overfills are
    rejected unless ``allow_overfill`` is True.
    """

    allow_overfill: bool = False
    _seen_events: set[str] = field(default_factory=set)
    _fills: list[Fill] = field(default_factory=list)

    def seen(self, event_id: str) -> bool:
        return event_id in self._seen_events

    def apply_fill(
        self,
        order: Order,
        *,
        fill_qty: float,
        fill_price: float,
        event_id: str,
        audit: AuditLog | None = None,
        venue_exec_id: str | None = None,
        liquidity_flag: str | None = None,
        fees: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Order, Fill | None, bool]:
        """Apply a fill. Returns ``(order, fill_or_none, applied)``.

        If ``event_id`` was already processed, returns ``(order, None, False)``
        without mutating quantity — idempotent guarantee.
        """
        if not event_id:
            raise ValidationError("event_id is required for idempotent fills", code="FILL_EVENT_ID_REQUIRED")
        if event_id in self._seen_events:
            if audit is not None:
                audit.append(
                    "fill_idempotent_skip",
                    f"duplicate fill event_id={event_id}",
                    order_id=order.order_id,
                    details={"event_id": event_id},
                )
            return order, None, False

        qty = float(fill_qty)
        px = float(fill_price)
        if qty <= 0:
            raise ValidationError("fill_qty must be positive", code="FILL_QTY_INVALID")
        if px <= 0:
            raise ValidationError("fill_price must be positive", code="FILL_PRICE_INVALID")

        residual = order.residual_qty
        if not self.allow_overfill and qty > residual + 1e-9:
            raise ExecutionError(
                f"fill_qty {qty} exceeds residual {residual}",
                code="FILL_OVERFILL",
                details={"fill_qty": qty, "residual": residual, "order_id": order.order_id},
            )

        applied_qty = min(qty, residual) if not self.allow_overfill else qty
        prev_filled = float(order.filled_qty)
        prev_avg = order.avg_fill_price
        new_filled = prev_filled + applied_qty
        if prev_avg is None or prev_filled <= 0:
            new_avg = px
        else:
            new_avg = ((prev_avg * prev_filled) + (px * applied_qty)) / new_filled

        order.filled_qty = new_filled
        order.avg_fill_price = float(new_avg)
        order.touch()

        fill = Fill(
            order_id=order.order_id,
            fill_qty=applied_qty,
            fill_price=px,
            event_id=event_id,
            venue_exec_id=venue_exec_id,
            liquidity_flag=liquidity_flag,
            fees=float(fees),
            metadata=dict(metadata or {}),
        )
        self._seen_events.add(event_id)
        self._fills.append(fill)

        order.audit.append(
            {
                "event": "fill",
                "event_id": event_id,
                "fill_qty": applied_qty,
                "fill_price": px,
                "filled_qty": order.filled_qty,
                "residual_qty": order.residual_qty,
                "avg_fill_price": order.avg_fill_price,
            }
        )
        if audit is not None:
            audit.append(
                "fill",
                f"filled {applied_qty} @ {px}",
                order_id=order.order_id,
                details=fill.to_dict(),
            )

        apply_fill_state(order, audit=audit)
        return order, fill, True

    def fills_for(self, order_id: str) -> list[Fill]:
        return [f for f in self._fills if f.order_id == order_id]

    def all_fills(self) -> list[Fill]:
        return list(self._fills)
