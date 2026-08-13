"""Cancel / replace (amend) helpers.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Never override hard risk limits.
- Replace creates a new order linked to the replaced one; idempotent via keys.
- Urgency NEVER overrides hard risk on the replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from iqrp.app.core.exceptions import ExecutionError
from iqrp.app.execution.order_manager.order import Order
from iqrp.app.execution.order_manager.order_lifecycle import (
    is_cancellable,
    is_replaceable,
    mark_replaced,
    request_cancel,
)
from iqrp.app.execution.order_manager.order_state import OrderState
from iqrp.app.execution.types import OrderType, TimeInForce, Urgency

if TYPE_CHECKING:
    from iqrp.app.execution.order_manager.audit import AuditLog


@dataclass(slots=True)
class CancelRequest:
    order_id: str
    reason: str = ""
    request_id: str = ""
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            object.__setattr__(self, "request_id", f"cx_{uuid4().hex[:12]}")


@dataclass(slots=True)
class ReplaceRequest:
    order_id: str
    quantity: float | None = None
    price: float | None = None
    stop_price: float | None = None
    order_type: OrderType | None = None
    time_in_force: TimeInForce | None = None
    urgency: Urgency | None = None
    request_id: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.request_id:
            object.__setattr__(self, "request_id", f"rp_{uuid4().hex[:12]}")


def begin_cancel(order: Order, *, audit: AuditLog | None = None, reason: str = "") -> Order:
    if order.state in {OrderState.CREATED, OrderState.VALIDATING, OrderState.APPROVED}:
        from iqrp.app.execution.order_manager.order_lifecycle import mark_cancelled

        return mark_cancelled(order, audit=audit, reason=reason or "cancelled_pre_submit")
    if not is_cancellable(order) and order.state is not OrderState.CANCEL_PENDING:
        raise ExecutionError(
            f"order {order.order_id} not cancellable in state {order.state.value}",
            code="ORDER_NOT_CANCELLABLE",
            details={"order_id": order.order_id, "state": order.state.value},
        )
    if order.state is OrderState.CANCEL_PENDING:
        return order
    return request_cancel(order, audit=audit)


def build_replacement(
    original: Order,
    request: ReplaceRequest,
    *,
    audit: AuditLog | None = None,
) -> Order:
    """Mark ``original`` REPLACED and return a new order with amended fields.

    Residual quantity is used when reducing size after partial fills. Urgency
    on the replacement still NEVER overrides hard risk at validation.
    """
    if not is_replaceable(original):
        raise ExecutionError(
            f"order {original.order_id} not replaceable in state {original.state.value}",
            code="ORDER_NOT_REPLACEABLE",
            details={"order_id": original.order_id, "state": original.state.value},
        )

    new_qty = float(request.quantity) if request.quantity is not None else original.residual_qty
    if request.quantity is not None:
        # Quantity is total desired size; residual for new order accounts for fills
        new_qty = max(float(request.quantity) - float(original.filled_qty), 0.0)
    if new_qty <= 0:
        raise ExecutionError(
            "replacement quantity leaves nothing to work",
            code="REPLACE_QTY_INVALID",
            details={"order_id": original.order_id, "requested": request.quantity},
        )

    replacement = Order(
        instrument=original.instrument,
        side=original.side,
        quantity=new_qty,
        order_type=request.order_type or original.order_type,
        price=request.price if request.price is not None else original.price,
        stop_price=request.stop_price if request.stop_price is not None else original.stop_price,
        time_in_force=request.time_in_force or original.time_in_force,
        venue=original.venue,
        algo=original.algo,
        urgency=request.urgency or original.urgency,
        strategy_id=original.strategy_id,
        portfolio_id=original.portfolio_id,
        parent_id=original.parent_id,
        account_id=original.account_id,
        client_order_id=f"{original.client_order_id}_rpl_{request.request_id}",
        idempotency_key=f"replace|{original.order_id}|{request.request_id}",
        tags={**original.tags, "replaces": original.order_id},
        metadata={
            **original.metadata,
            "replaces": original.order_id,
            "replace_request_id": request.request_id,
            "reason": request.reason,
        },
    )
    mark_replaced(original, audit=audit)
    original.metadata["replaced_by"] = replacement.order_id
    if audit is not None:
        audit.append(
            "replace",
            f"replaced {original.order_id} -> {replacement.order_id}",
            order_id=original.order_id,
            details={
                "original_id": original.order_id,
                "replacement_id": replacement.order_id,
                "request_id": request.request_id,
            },
        )
    return replacement
