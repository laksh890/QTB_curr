"""Parent order abstraction for multi-child execution plans.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Urgency influences aggressiveness but NEVER overrides hard risk.
- Never override hard risk limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from iqrp.app.execution.order_manager.execution_state import ExecutionState
from iqrp.app.execution.order_manager.order import Order
from iqrp.app.execution.types import Side, Urgency


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ParentOrder:
    """Parent / program order that owns child working orders."""

    instrument: str
    side: Side
    quantity: float
    strategy_id: str | None = None
    portfolio_id: str | None = None
    urgency: Urgency = Urgency.NORMAL
    algo: str | None = None
    parent_id: str = field(default_factory=lambda: f"par_{uuid4().hex[:16]}")
    state: ExecutionState = ExecutionState.IDLE
    child_ids: list[str] = field(default_factory=list)
    filled_qty: float = 0.0
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.instrument = str(self.instrument).upper()
        self.quantity = float(self.quantity)
        if isinstance(self.side, str):
            self.side = Side(self.side)
        if isinstance(self.urgency, str):
            self.urgency = Urgency(self.urgency)

    @property
    def residual_qty(self) -> float:
        return max(float(self.quantity) - float(self.filled_qty), 0.0)

    def attach_child(self, child: Order) -> None:
        child.parent_id = self.parent_id
        if child.order_id not in self.child_ids:
            self.child_ids.append(child.order_id)
        self.updated_at = _utc_now()

    def sync_fills_from_children(self, children: list[Order]) -> None:
        total = sum(float(c.filled_qty) for c in children if c.order_id in self.child_ids)
        self.filled_qty = float(total)
        self.updated_at = _utc_now()
        if self.filled_qty <= 0:
            return
        if self.residual_qty <= 1e-12:
            self.state = ExecutionState.COMPLETED
        else:
            self.state = ExecutionState.PARTIALLY_EXECUTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "instrument": self.instrument,
            "side": self.side.value,
            "quantity": float(self.quantity),
            "filled_qty": float(self.filled_qty),
            "residual_qty": self.residual_qty,
            "strategy_id": self.strategy_id,
            "portfolio_id": self.portfolio_id,
            "urgency": self.urgency.value,
            "algo": self.algo,
            "state": self.state.value,
            "child_ids": list(self.child_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }
