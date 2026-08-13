"""Institutional Execution Order Manager package.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Never override hard risk limits.
- Urgency influences aggressiveness but NEVER overrides hard risk.
- Idempotent fills/events; no future information.
"""

from __future__ import annotations

from iqrp.app.execution.order_manager.audit import AuditEntry, AuditLog
from iqrp.app.execution.order_manager.cancel_replace import (
    CancelRequest,
    ReplaceRequest,
    begin_cancel,
    build_replacement,
)
from iqrp.app.execution.order_manager.child_order import create_child_order, slice_parent
from iqrp.app.execution.order_manager.execution_state import (
    ALLOWED_EXECUTION_TRANSITIONS,
    ExecutionState,
    assert_execution_transition,
    transition_execution,
)
from iqrp.app.execution.order_manager.fill_manager import Fill, FillManager
from iqrp.app.execution.order_manager.order import Order, OrderSpec, target_to_orders
from iqrp.app.execution.order_manager.order_group import GroupType, OrderGroup
from iqrp.app.execution.order_manager.order_manager import OrderManager
from iqrp.app.execution.order_manager.order_state import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    OrderState,
    assert_transition,
    can_transition,
    transition_order,
)
from iqrp.app.execution.order_manager.order_validator import (
    InstrumentMeta,
    OrderValidator,
    ValidationResult,
)
from iqrp.app.execution.order_manager.parent_order import ParentOrder
from iqrp.app.execution.order_manager.position_reconciliation import (
    DiffSeverity,
    PositionReconciler,
    PositionSnapshot,
    ReconciliationAlert,
    ReconciliationResult,
)
from iqrp.app.execution.types import KillSwitch

__all__ = [
    "ALLOWED_EXECUTION_TRANSITIONS",
    "ALLOWED_TRANSITIONS",
    "AuditEntry",
    "AuditLog",
    "CancelRequest",
    "DiffSeverity",
    "ExecutionState",
    "Fill",
    "FillManager",
    "GroupType",
    "InstrumentMeta",
    "KillSwitch",
    "Order",
    "OrderGroup",
    "OrderManager",
    "OrderSpec",
    "OrderState",
    "OrderValidator",
    "ParentOrder",
    "PositionReconciler",
    "PositionSnapshot",
    "ReconciliationAlert",
    "ReconciliationResult",
    "ReplaceRequest",
    "TERMINAL_STATES",
    "ValidationResult",
    "assert_execution_transition",
    "assert_transition",
    "begin_cancel",
    "build_replacement",
    "can_transition",
    "create_child_order",
    "slice_parent",
    "target_to_orders",
    "transition_execution",
    "transition_order",
]
