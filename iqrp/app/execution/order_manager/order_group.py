"""Order groups for baskets, pairs, and spreads.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Never override hard risk limits.
- Group urgency NEVER overrides hard risk on member orders.
- No future information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from iqrp.app.execution.order_manager.execution_state import ExecutionState
from iqrp.app.execution.order_manager.order import Order
from iqrp.app.execution.types import Urgency


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GroupType(str, Enum):
    BASKET = "BASKET"
    PAIR = "PAIR"
    SPREAD = "SPREAD"
    LIST = "LIST"


@dataclass
class OrderGroup:
    """Logical grouping of related orders (basket / pair / spread)."""

    name: str
    group_type: GroupType = GroupType.BASKET
    group_id: str = field(default_factory=lambda: f"grp_{uuid4().hex[:16]}")
    urgency: Urgency = Urgency.NORMAL
    strategy_id: str | None = None
    portfolio_id: str | None = None
    state: ExecutionState = ExecutionState.IDLE
    order_ids: list[str] = field(default_factory=list)
    legs: dict[str, dict[str, Any]] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.group_type, str):
            self.group_type = GroupType(self.group_type)
        if isinstance(self.urgency, str):
            self.urgency = Urgency(self.urgency)

    def add_order(self, order: Order, *, leg: str | None = None, ratio: float = 1.0) -> None:
        if order.order_id not in self.order_ids:
            self.order_ids.append(order.order_id)
        key = leg or order.instrument
        self.legs[key] = {
            "order_id": order.order_id,
            "instrument": order.instrument,
            "side": order.side.value,
            "quantity": float(order.quantity),
            "ratio": float(ratio),
        }
        order.tags["group_id"] = self.group_id
        order.tags["group_type"] = self.group_type.value
        if leg:
            order.tags["leg"] = leg
        self.updated_at = _utc_now()

    def sync_state(self, orders: list[Order]) -> ExecutionState:
        members = [o for o in orders if o.order_id in self.order_ids]
        if not members:
            return self.state
        filled = all(o.state.value == "FILLED" for o in members)
        any_fill = any(o.filled_qty > 0 for o in members)
        any_fail = any(o.state.value in {"FAILED", "REJECTED"} for o in members)
        all_cancel = all(o.state.value == "CANCELLED" for o in members)
        if filled:
            self.state = ExecutionState.COMPLETED
        elif all_cancel:
            self.state = ExecutionState.CANCELLED
        elif any_fail:
            self.state = ExecutionState.FAILED
        elif any_fill:
            self.state = ExecutionState.PARTIALLY_EXECUTED
        else:
            self.state = ExecutionState.EXECUTING
        self.updated_at = _utc_now()
        return self.state

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "group_type": self.group_type.value,
            "urgency": self.urgency.value,
            "strategy_id": self.strategy_id,
            "portfolio_id": self.portfolio_id,
            "state": self.state.value,
            "order_ids": list(self.order_ids),
            "legs": dict(self.legs),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }
