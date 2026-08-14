"""Order lifecycle transition helpers.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Never override hard risk limits.
- Idempotent event processing; no future information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from iqrp.app.execution.order_manager.order_state import (
    TERMINAL_STATES,
    OrderState,
    transition_order,
)

if TYPE_CHECKING:
    from iqrp.app.execution.order_manager.audit import AuditLog
    from iqrp.app.execution.order_manager.order import Order


def begin_validation(order: Order, *, audit: AuditLog | None = None) -> Order:
    return transition_order(order, OrderState.VALIDATING, audit=audit, reason="begin_validation")


def approve(order: Order, *, audit: AuditLog | None = None) -> Order:
    return transition_order(order, OrderState.APPROVED, audit=audit, reason="approved")


def mark_submitted(
    order: Order, *, venue: str | None = None, audit: AuditLog | None = None
) -> Order:
    if venue:
        order.venue = venue
    from iqrp.app.execution.order_manager.order import _utc_now

    order.submitted_at = _utc_now()
    return transition_order(
        order,
        OrderState.SUBMITTED,
        audit=audit,
        reason="submitted",
        details={"venue": venue},
    )


def mark_acknowledged(
    order: Order,
    *,
    venue_order_id: str | None = None,
    audit: AuditLog | None = None,
) -> Order:
    from iqrp.app.execution.order_manager.order import _utc_now

    if venue_order_id:
        order.venue_order_id = venue_order_id
    order.acknowledged_at = _utc_now()
    return transition_order(
        order,
        OrderState.ACKNOWLEDGED,
        audit=audit,
        reason="acknowledged",
        details={"venue_order_id": venue_order_id},
    )


def mark_partial(order: Order, *, audit: AuditLog | None = None) -> Order:
    return transition_order(order, OrderState.PARTIALLY_FILLED, audit=audit, reason="partial_fill")


def mark_filled(order: Order, *, audit: AuditLog | None = None) -> Order:
    from iqrp.app.execution.order_manager.order import _utc_now

    order.completed_at = _utc_now()
    return transition_order(order, OrderState.FILLED, audit=audit, reason="filled")


def request_cancel(order: Order, *, audit: AuditLog | None = None) -> Order:
    return transition_order(
        order, OrderState.CANCEL_PENDING, audit=audit, reason="cancel_requested"
    )


def mark_cancelled(order: Order, *, audit: AuditLog | None = None, reason: str = "") -> Order:
    from iqrp.app.execution.order_manager.order import _utc_now

    order.completed_at = _utc_now()
    return transition_order(order, OrderState.CANCELLED, audit=audit, reason=reason or "cancelled")


def mark_rejected(order: Order, *, reason: str, audit: AuditLog | None = None) -> Order:
    from iqrp.app.execution.order_manager.order import _utc_now

    order.reject_reason = reason
    order.completed_at = _utc_now()
    return transition_order(
        order, OrderState.REJECTED, audit=audit, reason=reason, details={"reject_reason": reason}
    )


def mark_expired(order: Order, *, audit: AuditLog | None = None) -> Order:
    from iqrp.app.execution.order_manager.order import _utc_now

    order.completed_at = _utc_now()
    return transition_order(order, OrderState.EXPIRED, audit=audit, reason="expired")


def mark_replaced(order: Order, *, audit: AuditLog | None = None) -> Order:
    from iqrp.app.execution.order_manager.order import _utc_now

    order.completed_at = _utc_now()
    return transition_order(order, OrderState.REPLACED, audit=audit, reason="replaced")


def mark_failed(order: Order, *, reason: str, audit: AuditLog | None = None) -> Order:
    from iqrp.app.execution.order_manager.order import _utc_now

    order.reject_reason = reason
    order.completed_at = _utc_now()
    return transition_order(order, OrderState.FAILED, audit=audit, reason=reason)


def state_after_fill(order: Order) -> OrderState:
    """Derive fill-related state from residual quantity (no future info)."""
    if order.filled_qty <= 0:
        return order.state
    if order.residual_qty <= 1e-12:
        return OrderState.FILLED
    return OrderState.PARTIALLY_FILLED


def apply_fill_state(order: Order, *, audit: AuditLog | None = None) -> Order:
    target = state_after_fill(order)
    if target == order.state:
        return order
    if target is OrderState.FILLED:
        return mark_filled(order, audit=audit)
    if target is OrderState.PARTIALLY_FILLED:
        return mark_partial(order, audit=audit)
    return order


def is_cancellable(order: Order) -> bool:
    return order.state not in TERMINAL_STATES and order.state not in {
        OrderState.CANCEL_PENDING,
        OrderState.CREATED,
        OrderState.VALIDATING,
    }


def is_replaceable(order: Order) -> bool:
    return order.state in {
        OrderState.ACKNOWLEDGED,
        OrderState.PARTIALLY_FILLED,
        OrderState.SUBMITTED,
    }
