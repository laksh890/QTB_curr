"""Order lifecycle: create→validate→submit→ack→partial→fill; cancel/replace; idempotency."""

from __future__ import annotations

import pytest

from iqrp.app.core.exceptions import ExecutionError, ValidationError
from iqrp.app.execution.order_manager.order_state import OrderState
from iqrp.app.execution.types import Side, Urgency


def _approve_submit_ack(om, order):
    om.validate_and_approve(order.order_id)
    om.submit(order.order_id)
    om.acknowledge(order.order_id, venue_order_id="V-1", event_id=f"ack|{order.order_id}")
    return om.get(order.order_id)


def test_full_lifecycle_partial_then_fill(order_manager, make_limit_order):
    om = order_manager
    order = make_limit_order(quantity=100.0, price=100.0)
    assert order.state is OrderState.CREATED

    om.validate_and_approve(order.order_id)
    assert om.get(order.order_id).state is OrderState.APPROVED

    om.submit(order.order_id, venue="SIM", event_id="sub-1")
    assert om.get(order.order_id).state is OrderState.SUBMITTED

    om.acknowledge(order.order_id, venue_order_id="VO-1", event_id="ack-1")
    assert om.get(order.order_id).state is OrderState.ACKNOWLEDGED

    om.apply_fill(
        order.order_id,
        fill_qty=40.0,
        fill_price=100.01,
        event_id="fill-1",
    )
    o = om.get(order.order_id)
    assert o.state is OrderState.PARTIALLY_FILLED
    assert o.filled_qty == 40.0
    assert abs(o.residual_qty - 60.0) < 1e-9

    om.apply_fill(
        order.order_id,
        fill_qty=60.0,
        fill_price=100.02,
        event_id="fill-2",
    )
    o = om.get(order.order_id)
    assert o.state is OrderState.FILLED
    assert o.filled_qty == 100.0
    assert o.residual_qty == 0.0
    assert o.avg_fill_price is not None
    assert abs(o.avg_fill_price - (40 * 100.01 + 60 * 100.02) / 100) < 1e-9


def test_idempotent_fill_same_event_id(order_manager, make_limit_order):
    om = order_manager
    order = make_limit_order(quantity=50.0)
    _approve_submit_ack(om, order)

    om.apply_fill(order.order_id, fill_qty=20.0, fill_price=100.0, event_id="dup-fill")
    om.apply_fill(order.order_id, fill_qty=20.0, fill_price=100.0, event_id="dup-fill")
    o = om.get(order.order_id)
    assert o.filled_qty == 20.0
    assert o.state is OrderState.PARTIALLY_FILLED
    assert len(om.fills.fills_for(order.order_id)) == 1


def test_idempotent_create_key(order_manager):
    a = order_manager.create_order(
        instrument="AAPL",
        side=Side.BUY,
        quantity=10,
        order_type="LIMIT",
        price=100.0,
        idempotency_key="idem-1",
    )
    b = order_manager.create_order(
        instrument="AAPL",
        side=Side.BUY,
        quantity=10,
        order_type="LIMIT",
        price=100.0,
        idempotency_key="idem-1",
    )
    assert a.order_id == b.order_id
    assert len(order_manager.list_orders()) == 1


def test_idempotent_submit_ack_cancel_events(order_manager, make_limit_order):
    om = order_manager
    order = make_limit_order()
    om.validate_and_approve(order.order_id)
    om.submit(order.order_id, event_id="sub-x")
    om.submit(order.order_id, event_id="sub-x")  # duplicate
    assert om.get(order.order_id).state is OrderState.SUBMITTED

    om.acknowledge(order.order_id, event_id="ack-x")
    om.acknowledge(order.order_id, event_id="ack-x")
    assert om.get(order.order_id).state is OrderState.ACKNOWLEDGED

    om.cancel(order.order_id, reason="user", event_id="cx-1")
    om.cancel(order.order_id, reason="user", event_id="cx-1")
    assert om.get(order.order_id).state is OrderState.CANCELLED


def test_cancel_open_order(order_manager, make_limit_order):
    om = order_manager
    order = make_limit_order()
    _approve_submit_ack(om, order)
    om.cancel(order.order_id, reason="done")
    assert om.get(order.order_id).state is OrderState.CANCELLED


def test_replace_creates_new_order(order_manager, make_limit_order):
    om = order_manager
    order = make_limit_order(quantity=100.0, price=100.0)
    _approve_submit_ack(om, order)
    replacement = om.replace(
        order.order_id,
        quantity=80.0,
        price=99.5,
        reason="resize",
        request_id="rp-1",
        auto_approve=True,
    )
    assert replacement.order_id != order.order_id
    assert replacement.quantity == 80.0
    assert replacement.price == 99.5
    assert om.get(order.order_id).state is OrderState.REPLACED
    # Idempotent replace request
    again = om.replace(
        order.order_id,
        quantity=80.0,
        price=99.5,
        request_id="rp-1",
    )
    assert again.order_id == replacement.order_id


def test_invalid_transition_submit_from_created(order_manager, make_limit_order):
    order = make_limit_order()
    with pytest.raises(ExecutionError) as ei:
        order_manager.submit(order.order_id)
    assert ei.value.code == "ORDER_SUBMIT_BAD_STATE"


def test_invalid_ack_from_approved(order_manager, make_limit_order):
    order = make_limit_order()
    order_manager.validate_and_approve(order.order_id)
    with pytest.raises(ExecutionError) as ei:
        order_manager.acknowledge(order.order_id)
    assert ei.value.code == "ORDER_ACK_BAD_STATE"


def test_fill_bad_state_raises(order_manager, make_limit_order):
    order = make_limit_order()
    with pytest.raises(ExecutionError) as ei:
        order_manager.apply_fill(
            order.order_id, fill_qty=1.0, fill_price=100.0, event_id="f1"
        )
    assert ei.value.code == "ORDER_FILL_BAD_STATE"


def test_overfill_rejected(order_manager, make_limit_order):
    om = order_manager
    order = make_limit_order(quantity=10.0)
    _approve_submit_ack(om, order)
    with pytest.raises(ExecutionError) as ei:
        om.apply_fill(order.order_id, fill_qty=11.0, fill_price=100.0, event_id="of1")
    assert ei.value.code == "FILL_OVERFILL"


def test_process_event_dispatcher(order_manager, make_limit_order):
    om = order_manager
    order = make_limit_order(quantity=20.0)
    om.validate_and_approve(order.order_id)
    om.submit(order.order_id)
    om.process_event("e-ack", "acknowledge", order_id=order.order_id, payload={"venue_order_id": "V"})
    om.process_event(
        "e-fill",
        "fill",
        order_id=order.order_id,
        payload={"fill_qty": 20.0, "fill_price": 100.0},
    )
    assert om.get(order.order_id).state is OrderState.FILLED
    # duplicate process_event
    om.process_event("e-fill", "fill", order_id=order.order_id, payload={"fill_qty": 20.0, "fill_price": 100.0})
    assert om.get(order.order_id).filled_qty == 20.0


def test_process_event_reject_and_unknown(order_manager, make_limit_order):
    om = order_manager
    order = make_limit_order()
    om.validate_and_approve(order.order_id)
    om.submit(order.order_id)
    om.process_event("rej-1", "reject", order_id=order.order_id, payload={"reason": "venue"})
    assert om.get(order.order_id).state is OrderState.REJECTED
    with pytest.raises(ExecutionError):
        om.process_event("u-1", "weird", order_id=order.order_id)


def test_validation_reject_marks_rejected(order_manager):
    order = order_manager.create_order(
        instrument="AAPL",
        side=Side.BUY,
        quantity=0.5,  # below min_qty 1.0
        order_type="LIMIT",
        price=100.0,
    )
    with pytest.raises(ValidationError):
        order_manager.validate_and_approve(order.order_id)
    assert order_manager.get(order.order_id).state is OrderState.REJECTED


def test_list_orders_filter_and_unknown(order_manager, make_limit_order):
    a = make_limit_order(quantity=10)
    make_limit_order(quantity=20)
    order_manager.validate_and_approve(a.order_id)
    approved = order_manager.list_orders(state=OrderState.APPROVED)
    assert len(approved) == 1
    with pytest.raises(ExecutionError) as ei:
        order_manager.get("missing")
    assert ei.value.code == "ORDER_NOT_FOUND"


def test_create_from_target(order_manager):
    orders = order_manager.create_from_target(
        {"AAPL": 0.0},
        {"AAPL": 50.0},
        prices={"AAPL": 100.0},
        urgency=Urgency.HIGH,
    )
    assert len(orders) == 1
    assert orders[0].side is Side.BUY
    assert orders[0].quantity == 50.0


def test_urgency_does_not_bypass_fill_idempotency(order_manager, make_limit_order):
    """Architectural: duplicate event_id never double-counts regardless of urgency."""
    om = order_manager
    order = make_limit_order(quantity=100.0, urgency=Urgency.CRITICAL)
    _approve_submit_ack(om, order)
    om.apply_fill(order.order_id, fill_qty=50.0, fill_price=100.0, event_id="same")
    om.apply_fill(order.order_id, fill_qty=50.0, fill_price=101.0, event_id="same")
    assert om.get(order.order_id).filled_qty == 50.0
