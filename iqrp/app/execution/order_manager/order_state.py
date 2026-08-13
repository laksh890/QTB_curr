"""Order state machine with allowed transitions and audit helpers.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Never override hard risk limits.
- State transitions are audited; illegal transitions are rejected.
- No future information.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from iqrp.app.core.exceptions import ExecutionError

if TYPE_CHECKING:
    from iqrp.app.execution.order_manager.audit import AuditLog
    from iqrp.app.execution.order_manager.order import Order


class OrderState(str, Enum):
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    APPROVED = "APPROVED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    REPLACED = "REPLACED"
    FAILED = "FAILED"


# Allowed directed transitions. Terminal states have empty outgoing sets.
ALLOWED_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.CREATED: frozenset(
        {
            OrderState.VALIDATING,
            OrderState.REJECTED,
            OrderState.CANCELLED,
            OrderState.FAILED,
        }
    ),
    OrderState.VALIDATING: frozenset(
        {
            OrderState.APPROVED,
            OrderState.REJECTED,
            OrderState.FAILED,
            OrderState.CANCELLED,
        }
    ),
    OrderState.APPROVED: frozenset(
        {
            OrderState.SUBMITTED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.FAILED,
        }
    ),
    OrderState.SUBMITTED: frozenset(
        {
            OrderState.ACKNOWLEDGED,
            OrderState.REJECTED,
            OrderState.CANCEL_PENDING,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.FAILED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
        }
    ),
    OrderState.ACKNOWLEDGED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_PENDING,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.REPLACED,
            OrderState.FAILED,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_PENDING,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.REPLACED,
            OrderState.FAILED,
        }
    ),
    OrderState.FILLED: frozenset(),
    OrderState.CANCEL_PENDING: frozenset(
        {
            OrderState.CANCELLED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.FAILED,
            OrderState.REJECTED,
        }
    ),
    OrderState.CANCELLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.EXPIRED: frozenset(),
    OrderState.REPLACED: frozenset(),
    OrderState.FAILED: frozenset(),
}

TERMINAL_STATES: frozenset[OrderState] = frozenset(
    {
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
        OrderState.REPLACED,
        OrderState.FAILED,
    }
)


def can_transition(current: OrderState, new_state: OrderState) -> bool:
    if current == new_state and current is OrderState.PARTIALLY_FILLED:
        return True
    return new_state in ALLOWED_TRANSITIONS.get(current, frozenset())


def assert_transition(current: OrderState, new_state: OrderState) -> None:
    if not can_transition(current, new_state):
        raise ExecutionError(
            f"Illegal order state transition: {current.value} -> {new_state.value}",
            code="ORDER_STATE_TRANSITION_ILLEGAL",
            details={"from": current.value, "to": new_state.value},
        )


def transition_order(
    order: Order,
    new_state: OrderState,
    *,
    audit: AuditLog | None = None,
    reason: str = "",
    actor: str = "system",
    details: dict | None = None,
) -> Order:
    """Transition ``order`` to ``new_state`` with audit trail.

    Hard rule: illegal transitions are rejected; never silently coerced.
    """
    assert_transition(order.state, new_state)
    old = order.state
    order.state = new_state
    order.touch()
    payload = {"from": old.value, "to": new_state.value, "reason": reason}
    if details:
        payload.update(details)
    order.audit.append(
        {
            "event": "state_transition",
            "from": old.value,
            "to": new_state.value,
            "reason": reason,
            "actor": actor,
        }
    )
    if audit is not None:
        audit.append(
            "state_transition",
            reason or f"{old.value} -> {new_state.value}",
            order_id=order.order_id,
            actor=actor,
            details=payload,
        )
    return order
