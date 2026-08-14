"""Child order helpers for parent/program execution.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Never override hard risk limits.
- Urgency influences aggressiveness but NEVER overrides hard risk.
"""

from __future__ import annotations

from typing import Any

from iqrp.app.execution.order_manager.order import Order
from iqrp.app.execution.order_manager.parent_order import ParentOrder
from iqrp.app.execution.types import OrderType, Side, TimeInForce, Urgency


def create_child_order(
    parent: ParentOrder,
    *,
    quantity: float,
    order_type: OrderType = OrderType.LIMIT,
    price: float | None = None,
    venue: str | None = None,
    algo: str | None = None,
    time_in_force: TimeInForce = TimeInForce.DAY,
    urgency: Urgency | None = None,
    tags: dict[str, Any] | None = None,
) -> Order:
    """Create a child order linked to ``parent``.

    Child inherits instrument/side/strategy from parent. Urgency defaults to
    parent urgency but still NEVER overrides hard risk at validation/submit.
    """
    if quantity <= 0:
        raise ValueError("child quantity must be positive")
    if quantity > parent.residual_qty + 1e-9:
        raise ValueError(f"child quantity {quantity} exceeds parent residual {parent.residual_qty}")

    child = Order(
        instrument=parent.instrument,
        side=parent.side,
        quantity=float(quantity),
        order_type=order_type,
        price=price,
        time_in_force=time_in_force,
        venue=venue,
        algo=algo or parent.algo,
        urgency=urgency or parent.urgency,
        strategy_id=parent.strategy_id,
        portfolio_id=parent.portfolio_id,
        parent_id=parent.parent_id,
        tags=dict(tags or {}),
        metadata={"role": "child"},
    )
    parent.attach_child(child)
    return child


def slice_parent(
    parent: ParentOrder,
    *,
    slice_qty: float,
    n_slices: int | None = None,
    order_type: OrderType = OrderType.LIMIT,
    price: float | None = None,
    venue: str | None = None,
) -> list[Order]:
    """Slice remaining parent quantity into child orders of ``slice_qty``.

    Uses only current residual — no future information.
    """
    if slice_qty <= 0 and not n_slices:
        raise ValueError("slice_qty or n_slices required")

    remaining = parent.residual_qty
    children: list[Order] = []
    if n_slices is not None:
        if n_slices <= 0:
            raise ValueError("n_slices must be positive")
        base = remaining / n_slices
        for i in range(n_slices):
            qty = base if i < n_slices - 1 else remaining - base * (n_slices - 1)
            if qty <= 1e-12:
                continue
            children.append(
                create_child_order(
                    parent,
                    quantity=float(qty),
                    order_type=order_type,
                    price=price,
                    venue=venue,
                    tags={"slice_index": i},
                )
            )
        return children

    idx = 0
    while remaining > 1e-12:
        qty = min(slice_qty, remaining)
        children.append(
            create_child_order(
                parent,
                quantity=float(qty),
                order_type=order_type,
                price=price,
                venue=venue,
                tags={"slice_index": idx},
            )
        )
        remaining -= qty
        idx += 1
    return children


def is_child(order: Order) -> bool:
    return order.parent_id is not None


def child_side_matches_parent(child: Order, parent: ParentOrder) -> bool:
    return child.side == parent.side and isinstance(child.side, Side)
