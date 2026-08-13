"""Smart routing: single/multi venue; unavailable reject; kill switch; fallback; SimulatedVenue."""

from __future__ import annotations

import pytest

from iqrp.app.execution.order_manager.order import Order
from iqrp.app.execution.smart_routing import (
    SmartRouter,
    SimulatedVenue,
    Venue,
    VenueOrderRequest,
    VenueResponseStatus,
    VenueState,
    allocate_quantity,
    as_venue,
    build_fallback_chain,
    select_fallback,
)
from iqrp.app.execution.smart_routing.allocation import VenueAllocation
from iqrp.app.execution.smart_routing.scoring import VenueScore, rank_venues, score_venue
from iqrp.app.execution.types import KillSwitch, OrderType, Side, Urgency


def _order(qty: float = 100.0, otype: OrderType = OrderType.MARKET, price: float | None = None) -> Order:
    return Order(
        instrument="AAPL",
        side=Side.BUY,
        quantity=qty,
        order_type=otype,
        price=price,
        urgency=Urgency.NORMAL,
    )


def test_single_venue_route(smart_router, simulated_venue):
    decision = smart_router.route(_order(), [simulated_venue])
    assert decision.accepted
    assert decision.primary_venue_id == "SIM"
    assert decision.to_dict()["accepted"] is True


def test_multi_venue_allocation(multi_venues, kill_switch):
    router = SmartRouter(kill_switch=kill_switch, mode="multi", max_venues=2)
    decision = router.route(_order(qty=1000), multi_venues, mode="multi")
    assert decision.accepted
    assert len(decision.allocations) >= 1
    assert sum(a.quantity for a in decision.allocations) <= 1000 + 1e-6


def test_unavailable_venue_reject(kill_switch, market_context):
    bad = SimulatedVenue(
        venue_id="DOWN",
        instruments={"AAPL"},
        mode="fill",
        mid=100.0,
        spread=0.02,
    )
    bad.get_state().available = False
    router = SmartRouter(kill_switch=kill_switch)
    decision = router.route(_order(), [bad])
    assert not decision.accepted
    assert any(r.code in {"venue_unavailable", "no_venues"} or "unavailable" in r.message for r in decision.rejections) or decision.rejections


def test_halted_and_instrument_unavailable(kill_switch):
    v = SimulatedVenue(venue_id="H", instruments={"MSFT"}, mode="fill", mid=100.0)
    router = SmartRouter(kill_switch=kill_switch)
    d = router.route(_order(), [v])
    assert not d.accepted

    v2 = SimulatedVenue(venue_id="H2", instruments={"AAPL"}, mode="fill", mid=100.0)
    v2.get_state().halted = True
    assert not router.route(_order(), [v2]).accepted


def test_kill_switch_blocks_routing(smart_router, simulated_venue, kill_switch):
    kill_switch.engage_global("stop")
    # CRITICAL urgency must still be blocked
    o = Order(
        instrument="AAPL",
        side=Side.BUY,
        quantity=100.0,
        order_type=OrderType.MARKET,
        urgency=Urgency.CRITICAL,
    )
    d = smart_router.route(o, [simulated_venue])
    assert not d.accepted
    assert any(r.code == "kill_switch" for r in d.rejections)


def test_venue_kill_switch_and_router_flag(simulated_venue):
    router = SmartRouter(global_kill_switch=True)
    assert not router.route(_order(), [simulated_venue]).accepted
    router.set_kill_switch(False)
    assert router.route(_order(), [simulated_venue]).accepted

    simulated_venue.get_state().kill_switch = True
    assert not router.route(_order(), [simulated_venue]).accepted


def test_risk_check_blocks_venue(kill_switch, simulated_venue):
    router = SmartRouter(
        kill_switch=kill_switch,
        risk_check=lambda order, venue: (False, "risk_reject"),
    )
    d = router.route(_order(), [simulated_venue])
    assert not d.accepted


def test_fallback_chain(multi_venues, kill_switch):
    router = SmartRouter(kill_switch=kill_switch, mode="single", max_fallbacks=3)
    d = router.route(_order(), multi_venues)
    assert d.accepted
    assert d.fallback is not None
    assert d.fallback.to_dict()
    venues_map = {v.venue_id: as_venue(v) for v in multi_venues}
    select_fallback(
        d.fallback,
        venues_map,
        failed_venue_id=d.primary_venue_id or "SIM_A",
    )
    assert d.fallback is not None

    residual = router.route_residual(
        _order(qty=50),
        multi_venues,
        exclude_venues=[d.primary_venue_id],
        residual_qty=50,
    )
    assert residual.quantity == 50


def test_simulated_venue_modes(market_context):
    mid = market_context["mid"]
    spread = market_context["spread"]
    req = VenueOrderRequest(
        instrument="AAPL",
        side=Side.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        client_order_id="c1",
        order_id="o1",
    )
    for mode, status in [
        ("fill", VenueResponseStatus.FILL),
        ("partial", VenueResponseStatus.PARTIAL_FILL),
        ("ack", VenueResponseStatus.ACK),
        ("reject", VenueResponseStatus.REJECT),
    ]:
        v = SimulatedVenue(
            venue_id=f"M_{mode}",
            instruments={"AAPL"},
            mode=mode,
            mid=mid,
            spread=spread,
            partial_fraction=0.4,
        )
        resp = v.submit(req)
        assert resp.status is status
        if mode == "fill":
            assert resp.filled_qty == 100
        if mode == "partial":
            assert abs(resp.filled_qty - 40.0) < 1e-9
        if mode == "ack":
            cancel = v.cancel(resp.venue_order_id)
            assert cancel.status is VenueResponseStatus.CANCELLED
        if mode == "reject":
            assert resp.reason


def test_simulated_venue_limit_price_and_not_routable():
    v = SimulatedVenue(venue_id="L", instruments={"AAPL"}, mode="fill", mid=100.0, spread=0.02)
    req = VenueOrderRequest(
        instrument="AAPL",
        side=Side.SELL,
        quantity=10,
        order_type=OrderType.LIMIT,
        price=99.5,
    )
    resp = v.submit(req)
    assert resp.fill_price == 99.5

    v.get_state().available = False
    resp2 = v.submit(req)
    assert resp2.status is VenueResponseStatus.REJECT


def test_as_venue_from_dict_and_interface(simulated_venue):
    v = as_venue(
        {
            "venue_id": "D1",
            "mid": 100.0,
            "bid": 99.99,
            "ask": 100.01,
            "available_qty": 1e6,
            "instruments": ["AAPL"],
            "preference": 1.2,
        }
    )
    assert isinstance(v, Venue)
    assert v.venue_id == "D1"
    v.state.ensure_quotes()
    assert v.venue_state.mid is not None
    assert as_venue(simulated_venue).venue_id == "SIM"
    assert as_venue(v) is v


def test_invalid_order_rejected_by_router(smart_router, simulated_venue):
    bad = Order(instrument="", side=Side.BUY, quantity=0, order_type=OrderType.MARKET)
    assert not smart_router.route(bad, [simulated_venue]).accepted
    priced = Order(
        instrument="AAPL",
        side=Side.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        price=None,
    )
    assert not smart_router.route(priced, [simulated_venue]).accepted


def test_engine_route_helper(engine, simulated_venue):
    order = Order(instrument="AAPL", side=Side.BUY, quantity=10, order_type=OrderType.MARKET)
    d = engine.route(order, [simulated_venue])
    assert d.accepted
