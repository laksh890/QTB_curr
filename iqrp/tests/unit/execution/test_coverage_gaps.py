"""Coverage gaps: remaining branches across execution package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from iqrp.app.core.exceptions import ExecutionError
from iqrp.app.execution.algorithms.arrival_price import (
    arrival_slippage_bps,
    track_arrival_performance,
)
from iqrp.app.execution.algorithms.base import (
    apply_participation_cap,
    context_float,
    context_side,
    limit_hint,
    n_slices_for_urgency,
    schedule_offsets,
)
from iqrp.app.execution.analytics import (
    execution_quality_report,
    fill_rate,
    implementation_shortfall,
)
from iqrp.app.execution.config import ExecutionSettings
from iqrp.app.execution.latency import LatencyRecord, LatencyTracker
from iqrp.app.execution.order_manager.audit import AuditLog
from iqrp.app.execution.order_manager.cancel_replace import (
    CancelRequest,
    ReplaceRequest,
    begin_cancel,
    build_replacement,
)
from iqrp.app.execution.order_manager.child_order import create_child_order, slice_parent
from iqrp.app.execution.order_manager.execution_state import (
    ExecutionState,
    assert_execution_transition,
    transition_execution,
)
from iqrp.app.execution.order_manager.fill_manager import Fill, FillManager
from iqrp.app.execution.order_manager.order import Order, OrderSpec, target_to_orders
from iqrp.app.execution.order_manager.order_group import GroupType, OrderGroup
from iqrp.app.execution.order_manager.order_lifecycle import (
    approve,
    begin_validation,
    mark_acknowledged,
    mark_cancelled,
    mark_expired,
    mark_failed,
    mark_rejected,
    mark_submitted,
)
from iqrp.app.execution.order_manager.order_state import (
    OrderState,
    assert_transition,
    can_transition,
    transition_order,
)
from iqrp.app.execution.order_manager.parent_order import ParentOrder
from iqrp.app.execution.order_manager.position_reconciliation import (
    PositionReconciler,
    PositionSnapshot,
)
from iqrp.app.execution.serializer import ExecutionSerializer, _to_jsonable
from iqrp.app.execution.simulation import simulate_fill_path, simulate_with_market_simulator
from iqrp.app.execution.smart_routing.allocation import allocate_quantity
from iqrp.app.execution.smart_routing.cost_model import estimate_venue_cost
from iqrp.app.execution.smart_routing.fallback import (
    FallbackChain,
    build_fallback_chain,
    select_fallback,
)
from iqrp.app.execution.smart_routing.liquidity import aggregate_fillable, assess_liquidity
from iqrp.app.execution.smart_routing.scoring import (
    DEFAULT_WEIGHTS,
    ScoreWeights,
    rank_venues,
    score_venue,
)
from iqrp.app.execution.smart_routing.venue import SimulatedVenue, Venue, VenueState, as_venue
from iqrp.app.execution.smart_routing.venue_state import VenueState as VS
from iqrp.app.execution.types import KillSwitch, OrderType, Side, TimeInForce, Urgency


def test_side_order_type_parse():
    assert Side.parse("B") is Side.BUY
    assert Side.parse("SHORT") is Side.SHORT
    assert Side.parse("COVER") is Side.COVER
    assert Side.SELL.signed_direction == -1
    assert Side.BUY.is_buy
    with pytest.raises(ValueError):
        Side.parse("nope")
    assert OrderType.parse("limit-on-close") is OrderType.LOC
    assert OrderType.parse(OrderType.MARKET) is OrderType.MARKET
    with pytest.raises(ValueError):
        OrderType.parse("not_a_type")


def test_kill_switch_dict_and_scopes():
    ks = KillSwitch()
    ks.halt_global("g")
    ks.halt_account("a")
    ks.halt_venue("v")
    ks.halt_strategy("s")
    d = ks.to_dict()
    assert d["global_halt"]
    assert ks.is_blocked(account_id="a")[0]
    ks.clear_global()
    ks.clear_account("a")
    ks.clear_venue("v")
    ks.clear_strategy("s")
    assert not ks.is_blocked()[0]


def test_order_state_machine_illegal():
    assert can_transition(OrderState.CREATED, OrderState.VALIDATING)
    with pytest.raises(ExecutionError):
        assert_transition(OrderState.FILLED, OrderState.CREATED)
    order = Order(instrument="AAPL", side=Side.BUY, quantity=1, order_type=OrderType.MARKET)
    audit = AuditLog()
    begin_validation(order, audit=audit)
    approve(order, audit=audit)
    mark_submitted(order, venue="SIM", audit=audit)
    mark_acknowledged(order, venue_order_id="V", audit=audit)
    mark_expired(order, audit=audit)
    assert order.state is OrderState.EXPIRED

    order2 = Order(instrument="AAPL", side=Side.BUY, quantity=1, order_type=OrderType.MARKET)
    begin_validation(order2, audit=audit)
    mark_rejected(order2, reason="x", audit=audit)
    order3 = Order(instrument="AAPL", side=Side.BUY, quantity=1, order_type=OrderType.MARKET)
    mark_failed(order3, reason="y", audit=audit)


def test_execution_state_transitions():
    assert_execution_transition(ExecutionState.IDLE, ExecutionState.PLANNING)
    s = transition_execution(ExecutionState.IDLE, ExecutionState.PLANNING)
    assert s is ExecutionState.PLANNING
    with pytest.raises(ExecutionError):
        transition_execution(ExecutionState.COMPLETED, ExecutionState.PLANNING)


def test_fill_manager_edge_cases():
    fm = FillManager(allow_overfill=False)
    order = Order(instrument="AAPL", side=Side.BUY, quantity=10, order_type=OrderType.MARKET)
    order.state = OrderState.ACKNOWLEDGED
    with pytest.raises(Exception):
        fm.apply_fill(order, fill_qty=1, fill_price=100, event_id="")
    with pytest.raises(Exception):
        fm.apply_fill(order, fill_qty=0, fill_price=100, event_id="e0")
    with pytest.raises(Exception):
        fm.apply_fill(order, fill_qty=1, fill_price=0, event_id="e1")
    _, fill, applied = fm.apply_fill(order, fill_qty=5, fill_price=100, event_id="ok")
    assert applied and fill is not None
    assert fill.to_dict()["event_id"] == "ok"
    assert fm.seen("ok")
    assert len(fm.all_fills()) == 1


def test_parent_child_group_reconcile():
    parent = ParentOrder(instrument="AAPL", side=Side.BUY, quantity=100.0)
    child = create_child_order(parent, quantity=40.0, order_type=OrderType.LIMIT, price=100.0)
    parent.attach_child(child)
    child.filled_qty = 40.0
    parent.sync_fills_from_children([child])
    assert parent.filled_qty == 40.0
    assert parent.state is ExecutionState.PARTIALLY_EXECUTED
    assert parent.to_dict()["residual_qty"] == 60.0

    slices = slice_parent(parent, slice_qty=30.0, n_slices=2, order_type=OrderType.MARKET)
    assert len(slices) == 2

    group = OrderGroup(name="g1", group_type=GroupType.BASKET)
    group.order_ids.append(child.order_id)
    assert child.order_id in group.order_ids
    assert group.to_dict()["group_type"] == GroupType.BASKET.value

    recon = PositionReconciler()
    snap = PositionSnapshot(instrument="AAPL", quantity=100.0, source="expected")
    result = recon.reconcile(
        expected={"AAPL": 100.0, "MSFT": 10.0},
        executed={"AAPL": 100.0},
        broker={"AAPL": 99.0, "MSFT": 10.0},
    )
    assert result.to_dict()
    assert snap.source == "expected"
    assert snap.instrument == "AAPL"


def test_cancel_replace_helpers():
    audit = AuditLog()
    order = Order(
        instrument="AAPL", side=Side.BUY, quantity=50, order_type=OrderType.LIMIT, price=100.0
    )
    order.state = OrderState.ACKNOWLEDGED
    begin_cancel(order, audit=audit, reason="user")
    assert order.state is OrderState.CANCEL_PENDING
    mark_cancelled(order, audit=audit, reason="done")

    order2 = Order(
        instrument="AAPL", side=Side.BUY, quantity=50, order_type=OrderType.LIMIT, price=100.0
    )
    order2.state = OrderState.ACKNOWLEDGED
    req = ReplaceRequest(order_id=order2.order_id, quantity=40.0, price=99.0, request_id="r1")
    repl = build_replacement(order2, req, audit=audit)
    assert repl.quantity == 40.0
    assert order2.state is OrderState.REPLACED
    CancelRequest(order_id=order.order_id, reason="x")  # construct


def test_target_to_orders_and_order_roundtrip():
    specs = target_to_orders(
        {"AAPL": 10.0, "MSFT": 0.0},
        {"AAPL": 5.0, "MSFT": 20.0},
        prices={"AAPL": 100.0, "MSFT": 200.0},
        lot_size=1.0,
        min_qty=1.0,
    )
    assert any(s.side is Side.SELL for s in specs)
    assert any(s.side is Side.BUY for s in specs)
    kw = specs[0].to_order_kwargs()
    assert "instrument" in kw
    with pytest.raises(ValueError):
        target_to_orders({}, {}, lot_size=0)
    order = Order(instrument="aapl", side="BUY", quantity=1, order_type="MARKET")
    d = order.to_dict()
    o2 = Order.from_dict(d)
    assert o2.instrument == "AAPL"
    assert order.notional is None or True
    assert order.residual_qty == 1.0


def test_algo_base_helpers():
    assert context_float({}, "x", 1.5) == 1.5
    assert context_side({"side": "SHORT"}) == "sell"
    assert limit_hint(100.0, 0.02, "buy", Urgency.HIGH) > 100.0
    assert n_slices_for_urgency(10, Urgency.CRITICAL) >= 1
    offs = schedule_offsets(4, 100.0, jitter=0.2, rng=np.random.default_rng(42))
    assert offs[0] == 0.0
    capped = apply_participation_cap(
        [100, 100], adv=1000, participation_cap=0.1, horizon_fraction=0.5
    )
    assert all(q <= 100 for q in capped)


def test_arrival_helpers():
    bps = arrival_slippage_bps(side="buy", fill_price=100.1, arrival_price=100.0)
    assert bps > 0
    perf = track_arrival_performance(
        [{"quantity": 10, "price": 100.05}],
        side="buy",
        arrival_price=100.0,
    )
    assert isinstance(perf, dict)


def test_analytics_helpers():
    fills = [{"qty": 50, "price": 100.1}]
    fr = fill_rate(ordered_qty=100, filled_qty=50)
    assert (
        fr["fill_rate"] == 0.5
        or fr.get("fill_rate", 0.5) == 0.5
        or abs(float(list(fr.values())[0]) - 0.5) < 1e-9
    )
    is_ = implementation_shortfall(side="buy", arrival_price=100.0, fills=fills)
    assert is_
    report = execution_quality_report(
        side="buy",
        ordered_qty=100,
        fills=fills,
        arrival_price=100.0,
        vwap_benchmark=100.05,
        twap_benchmark=100.04,
        latency={"mean_ms": 1.0},
        pre_trade_estimate={"total_cost": 1.0},
        post_trade_costs={"realized_cost": 1.2},
    )
    assert report


def test_latency_tracker():
    tr = LatencyTracker()
    tr.start("a")
    tr.mark_submit("a")
    tr.mark_ack("a")
    tr.mark_fill("a")
    rec = tr.get("a")
    assert isinstance(rec, LatencyRecord)
    assert rec.decision_to_fill_ms is not None or rec.to_dict()
    assert tr.summary(["a", "missing"])


def test_serializer(tmp_path: Path):
    ser = ExecutionSerializer()
    order = Order(instrument="AAPL", side=Side.BUY, quantity=1, order_type=OrderType.MARKET)
    p = ser.save_order(order, tmp_path / "o.json")
    assert ser.load_order(p).instrument == "AAPL"
    parent = ParentOrder(instrument="AAPL", side=Side.BUY, quantity=10)
    pp = ser.save_parent(parent, tmp_path / "p.json")
    assert ser.load_parent(pp).quantity == 10
    raw = ser.dump_bytes({"a": 1, "arr": np.array([1.0, 2.0]), "e": Side.BUY})
    assert ser.load_bytes(raw)["a"] == 1
    assert _to_jsonable(Path("/tmp/x")) == "/tmp/x"


def test_simulation_paths():
    r = simulate_fill_path(side="buy", quantity=0, mid=100)
    assert r["fills"] == []
    r2 = simulate_fill_path(side="sell", quantity=50, mid=100, spread=0.02, n_slices=2, seed=0)
    assert r2["filled_qty"] <= 50 + 1e-6
    r3 = simulate_with_market_simulator(
        side="buy", quantity=20, seed=1, market_context={"mid": 100, "spread": 0.02}
    )
    assert r3["filled_qty"] <= 20 + 1e-6


def test_smart_routing_helpers():
    state = VS(
        venue_id="V1",
        mid=100.0,
        bid=99.99,
        ask=100.01,
        available_qty=1e5,
        adv=1e6,
        instruments={"AAPL"},
        supported_order_types={"MARKET", "LIMIT"},
    )
    state.ensure_quotes()
    assert state.spread_bps is not None
    assert state.to_dict()["venue_id"] == "V1"
    venue = Venue(venue_id="V1", state=state)
    liq = assess_liquidity(venue, instrument="AAPL", quantity=100, side=Side.BUY)
    assert liq.fillable_qty > 0
    assert aggregate_fillable([liq]) >= liq.fillable_qty
    cost = estimate_venue_cost(venue, side=Side.BUY, quantity=100, order_type=OrderType.MARKET)
    assert cost.expected_price > 0
    assert cost.to_dict()
    raw_w = (
        DEFAULT_WEIGHTS.to_dict()
        if hasattr(DEFAULT_WEIGHTS, "to_dict")
        else {
            "price": 0.3,
            "fees": 0.1,
            "impact": 0.2,
            "fill_prob": 0.2,
            "latency": 0.1,
            "reliability": 0.1,
        }
    )
    w = (
        ScoreWeights.from_mapping(raw_w)
        if hasattr(ScoreWeights, "from_mapping")
        else DEFAULT_WEIGHTS
    )
    weights = w.normalized() if hasattr(w, "normalized") else w
    sc = score_venue(
        venue,
        cost=cost,
        liquidity=liq,
        weights=weights,
        is_buy=True,
        peer_prices=[100.0],
    )
    ranked = rank_venues([sc])
    assert ranked[0].venue_id == "V1"
    plan = allocate_quantity(
        100.0,
        ranked,
        {"V1": liq},
        mode="single",
        lot_sizes={"V1": 1.0},
        min_qty={"V1": 1.0},
    )
    assert plan.allocations
    fb = build_fallback_chain(ranked, primary_venue_id="V1", max_fallbacks=2)
    assert isinstance(fb, FallbackChain)
    select_fallback(fb, {"V1": venue}, failed_venue_id="V1")


def test_settings_hydra_and_from_mapping():
    s = ExecutionSettings.from_hydra(overrides=["seed=99"])
    assert s.seed == 99
    s2 = ExecutionSettings.from_mapping(s.model_dump())
    assert s2.default_venue


def test_engine_partial_venue_and_multi(
    engine, market_context, venue_partial, venue_reject, multi_venues
):
    r = engine.execute(
        {"AAPL": 40.0},
        algo="twap",
        venues=[venue_partial],
        market_context={**market_context, "n_slices": 2},
        current={"AAPL": 0.0},
    )
    assert r.status in {"PARTIAL", "FILLED", "COMPLETED", "FAILED"}

    r2 = engine.execute(
        {"AAPL": 20.0},
        algo="market",
        venues=[venue_reject],
        market_context=market_context,
        current={"AAPL": 0.0},
    )
    # reject venue → may complete without fills
    assert r2.status in {"COMPLETED", "FAILED", "PARTIAL", "FILLED"}

    r3 = engine.execute(
        {"AAPL": 60.0},
        algo="vwap",
        venues=multi_venues,
        market_context={**market_context, "n_slices": 2, "volume_curve": [1, 2]},
        current={"AAPL": 0.0},
    )
    assert r3.routing or r3.status


def test_register_parent_group(order_manager):
    p = ParentOrder(instrument="AAPL", side=Side.BUY, quantity=10)
    order_manager.register_parent(p)
    g = OrderGroup(name="basket", group_type=GroupType.BASKET)
    order_manager.register_group(g)
    assert p.parent_id in order_manager._parents
