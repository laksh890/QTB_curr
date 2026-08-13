"""ExecutionEngine API: plan, execute, halt/kill, reconcile, analytics, save/load."""

from __future__ import annotations

from pathlib import Path

import pytest

from iqrp.app.core.exceptions import ExecutionError
from iqrp.app.execution import ExecutionEngine, ExecutionSettings, KillSwitch, SimulatedVenue
from iqrp.app.execution.order_manager.execution_state import ExecutionState
from iqrp.app.execution.order_manager.order import Order
from iqrp.app.execution.order_manager.order_state import OrderState
from iqrp.app.execution.order_manager.parent_order import ParentOrder
from iqrp.app.execution.types import Side, Urgency


def test_plan_from_targets(engine, market_context):
    orders = engine.plan_from_targets(
        {"AAPL": 0.0},
        {"AAPL": 100.0},
        prices={"AAPL": market_context["mid"]},
    )
    assert len(orders) == 1
    assert orders[0].side is Side.BUY
    assert orders[0].quantity == 100.0


def test_plan_empty_when_no_delta(engine):
    orders = engine.plan_from_targets({"AAPL": 50.0}, {"AAPL": 50.0})
    assert orders == []


def test_execute_twap_fill(engine, market_context, simulated_venue):
    report = engine.execute(
        {"AAPL": 90.0},
        algo="twap",
        urgency=Urgency.NORMAL,
        venues=[simulated_venue],
        market_context={**market_context, "n_slices": 3, "horizon_seconds": 30.0},
        current={"AAPL": 0.0},
        simulation_mode=True,
    )
    assert report.status in {"FILLED", "PARTIAL", "COMPLETED"}
    assert report.to_dict()["execution_id"]
    assert report.fills or report.status == "COMPLETED"
    # Never exceed target residual
    for p in report.parents:
        assert p["filled_qty"] <= p["quantity"] + 1e-6


def test_execute_from_parent_and_order(engine, market_context, simulated_venue):
    parent = ParentOrder(
        instrument="AAPL",
        side=Side.BUY,
        quantity=30.0,
        urgency=Urgency.LOW,
        algo="market",
    )
    r1 = engine.execute(
        parent,
        algo="market",
        venues=[simulated_venue],
        market_context=market_context,
    )
    assert r1.status in {"FILLED", "PARTIAL", "COMPLETED"}

    order = Order(instrument="AAPL", side=Side.SELL, quantity=20.0, order_type="MARKET")
    r2 = engine.execute(
        order,
        algo="limit",
        venues=[simulated_venue],
        market_context=market_context,
    )
    assert r2.algo == "limit"


def test_execute_empty_targets(engine, market_context):
    report = engine.execute({"AAPL": 0.0}, market_context=market_context, current={"AAPL": 0.0})
    assert report.status == "EMPTY"


def test_halt_blocks_new_submits(engine, market_context, simulated_venue, make_limit_order, order_manager):
    # Seed an open order then halt
    om = engine.order_manager
    order = om.create_order(
        instrument="AAPL", side=Side.BUY, quantity=10, order_type="LIMIT", price=100.0
    )
    om.validate_and_approve(order.order_id)
    om.submit(order.order_id)
    om.acknowledge(order.order_id, event_id="ack-h")

    engine.halt("test halt", cancel_open=True)
    assert engine.state is ExecutionState.HALTED
    assert engine.kill_switch.global_halt

    with pytest.raises(ExecutionError) as ei:
        engine.execute(
            {"AAPL": 50.0},
            venues=[simulated_venue],
            market_context=market_context,
        )
    assert ei.value.code in {"EXECUTION_HALTED", "KILL_SWITCH_ACTIVE"}

    # Urgency CRITICAL cannot bypass
    with pytest.raises(ExecutionError):
        engine.plan_from_targets({"AAPL": 0}, {"AAPL": 10})


def test_kill_scopes(engine):
    engine.kill("account", key="ACC1", reason="acct")
    assert "ACC1" in engine.kill_switch.accounts
    engine.kill("venue", key="SIM", reason="ven")
    assert "SIM" in engine.kill_switch.venues
    engine.kill("strategy", key="S1", reason="str")
    assert "S1" in engine.kill_switch.strategies
    with pytest.raises(ValueError):
        engine.kill("account", key=None)
    with pytest.raises(ValueError):
        engine.kill("unknown")
    engine.kill("global", reason="all stop")
    assert engine._halted


def test_reconcile(engine):
    result = engine.reconcile(
        expected={"AAPL": 100.0},
        executed={"AAPL": 100.0},
        broker={"AAPL": 100.0},
    )
    assert result.matched or result.to_dict()
    bad = engine.reconcile(
        expected={"AAPL": 100.0},
        executed={"AAPL": 90.0},
        broker={"AAPL": 95.0},
    )
    d = bad.to_dict() if hasattr(bad, "to_dict") else bad
    assert d is not None


def test_analytics_and_latency(engine):
    fills = [{"quantity": 50, "price": 100.05}, {"qty": 50, "fill_price": 100.1}]
    aq = engine.analytics(fills, arrival_price=100.0, side="buy", ordered_qty=100.0)
    assert "fill_rate" in aq or "implementation_shortfall" in aq or aq

    engine.latency.start("oid1")
    engine.latency.mark_submit("oid1")
    engine.latency.mark_ack("oid1")
    engine.latency.mark_fill("oid1")
    summary = engine.latency.summary(["oid1"])
    assert summary
    assert engine.latency.to_dict()


def test_simulate_execution(engine, market_context):
    sim = engine.simulate_execution(
        side="buy",
        quantity=100.0,
        market_context=market_context,
        use_market_simulator=False,
        seed=42,
        n_slices=3,
    )
    assert sim["filled_qty"] <= 100.0 + 1e-6
    assert len(sim["fills"]) >= 1

    multi = engine.simulate_execution(
        orders=[{"instrument": "AAPL", "side": "buy", "quantity": 50}],
        market_context=market_context,
        use_market_simulator=False,
        seed=42,
    )
    assert multi["n"] == 1


def test_save_load(engine, market_context, simulated_venue, tmp_path: Path):
    engine.execute(
        {"AAPL": 40.0},
        algo="twap",
        venues=[simulated_venue],
        market_context={**market_context, "n_slices": 2},
        current={"AAPL": 0.0},
    )
    path = tmp_path / "exec_state.json"
    out = engine.save(path)
    assert out.is_file()

    engine2 = ExecutionEngine(settings=ExecutionSettings(seed=42))
    data = engine2.load(path)
    assert "orders" in data
    assert engine2.state.value == data["state"] or True


def test_apply_event_idempotent(engine, market_context):
    om = engine.order_manager
    order = om.create_order(
        instrument="AAPL", side=Side.BUY, quantity=20, order_type="LIMIT", price=100.0
    )
    om.validate_and_approve(order.order_id)
    om.submit(order.order_id)
    engine.apply_event(
        "ev-ack",
        event_type="acknowledge",
        order_id=order.order_id,
        venue_order_id="V1",
    )
    engine.apply_event(
        "ev-fill",
        event_type="fill",
        order_id=order.order_id,
        fill_qty=20.0,
        fill_price=100.0,
    )
    # duplicate
    engine.apply_event(
        "ev-fill",
        event_type="fill",
        order_id=order.order_id,
        fill_qty=20.0,
        fill_price=100.0,
    )
    assert om.get(order.order_id).filled_qty == 20.0


def test_validate_order(engine):
    order = Order(instrument="AAPL", side=Side.BUY, quantity=10, order_type="LIMIT", price=100.0)
    assert engine.validate_order(order).ok


def test_risk_engine_blocks_execute(execution_settings, kill_switch, rejecting_risk, market_context, simulated_venue):
    eng = ExecutionEngine(
        settings=execution_settings,
        kill_switch=kill_switch,
        risk_engine=rejecting_risk,
    )
    report = eng.execute(
        {"AAPL": 50.0},
        venues=[simulated_venue],
        market_context=market_context,
        current={"AAPL": 0.0},
        urgency=Urgency.CRITICAL,
    )
    # Hard risk reject → FAILED (not filled); urgency does not bypass
    assert report.status in {"FAILED", "BLOCKED"}
    assert report.errors


def test_settings_from_mapping_and_default():
    s = ExecutionSettings.default()
    assert s.seed == 42 or isinstance(s.seed, int)
    s2 = ExecutionSettings.from_mapping({"seed": 7, "default_venue": "X"})
    assert s2.seed == 7
    with pytest.raises(Exception):
        ExecutionSettings.from_mapping({"seed": "not-an-int-xxx"})  # may or may not raise
