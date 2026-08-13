"""Institutional Execution Platform.

CRITICAL RULES
--------------
- Execution never generates alpha or invents positions.
- Never exceed approved target residual.
- Risk Intelligence is authoritative when provided.
- Kill switches are fail-safe; urgency never overrides hard risk.
- Events/fills are idempotent; no future information.
"""

from __future__ import annotations

from iqrp.app.execution.config import ExecutionSettings
from iqrp.app.execution.engine import ExecutionEngine, ExecutionReport
from iqrp.app.execution.order_manager import (
    Fill,
    FillManager,
    Order,
    OrderManager,
    OrderValidator,
    ParentOrder,
    PositionReconciler,
    ReconciliationResult,
    ValidationResult,
)
from iqrp.app.execution.smart_routing import (
    RoutingDecision,
    SimulatedVenue,
    SmartRouter,
    Venue,
    VenueState,
)
from iqrp.app.execution.types import KillSwitch, OrderType, Side, TimeInForce, Urgency

__all__ = [
    "ExecutionEngine",
    "ExecutionReport",
    "ExecutionSettings",
    "Fill",
    "FillManager",
    "KillSwitch",
    "Order",
    "OrderManager",
    "OrderType",
    "OrderValidator",
    "ParentOrder",
    "PositionReconciler",
    "ReconciliationResult",
    "RoutingDecision",
    "Side",
    "SimulatedVenue",
    "SmartRouter",
    "TimeInForce",
    "Urgency",
    "ValidationResult",
    "Venue",
    "VenueState",
]
