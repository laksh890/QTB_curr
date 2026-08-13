"""Order validation: tick/lot/qty/price bands; risk reject; kill switch blocks submit."""

from __future__ import annotations

import pytest

from iqrp.app.core.exceptions import ExecutionError, ValidationError
from iqrp.app.execution.config import ExecutionSettings, PriceBandConfig, RiskConfig
from iqrp.app.execution.order_manager.order_manager import OrderManager
from iqrp.app.execution.order_manager.order_state import OrderState
from iqrp.app.execution.order_manager.order_validator import InstrumentMeta, OrderValidator
from iqrp.app.execution.types import KillSwitch, OrderType, Side, Urgency


def test_tick_size_reject(order_manager_with_meta):
    om = order_manager_with_meta
    order = om.create_order(
        instrument="AAPL",
        side=Side.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        price=100.005,  # not multiple of 0.01
    )
    result = om.validator.validate(order)
    assert not result.ok
    assert any("tick_size" in e for e in result.errors)


def test_lot_size_reject(execution_settings, kill_switch):
    meta = InstrumentMeta(symbol="AAPL", lot_size=10.0, tick_size=0.01, min_qty=10.0)
    om = OrderManager(
        execution_settings,
        kill_switch=kill_switch,
        validator=OrderValidator(execution_settings, instruments={"AAPL": meta}),
    )
    order = om.create_order(
        instrument="AAPL",
        side=Side.BUY,
        quantity=15,  # not multiple of 10
        order_type=OrderType.LIMIT,
        price=100.0,
    )
    result = om.validator.validate(order)
    assert not result.ok
    assert any("lot_size" in e for e in result.errors)


def test_qty_below_min_and_above_max(order_manager_with_meta):
    om = order_manager_with_meta
    low = om.create_order(
        instrument="AAPL", side=Side.BUY, quantity=0.5, order_type="LIMIT", price=100.0
    )
    high = om.create_order(
        instrument="AAPL", side=Side.BUY, quantity=200_000, order_type="LIMIT", price=100.0
    )
    assert not om.validator.validate(low).ok
    assert not om.validator.validate(high).ok


def test_price_band_reject(order_manager_with_meta):
    om = order_manager_with_meta
    order = om.create_order(
        instrument="AAPL",
        side=Side.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        price=130.0,  # 30% from ref 100
    )
    result = om.validator.validate(order)
    assert not result.ok
    assert any("band" in e.lower() for e in result.errors)


def test_limit_requires_price(order_manager):
    order = order_manager.create_order(
        instrument="AAPL",
        side=Side.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        price=None,
    )
    result = order_manager.validator.validate(order)
    assert not result.ok
    assert any("require price" in e for e in result.errors)


def test_stop_requires_stop_price(order_manager):
    order = order_manager.create_order(
        instrument="AAPL",
        side=Side.SELL,
        quantity=10,
        order_type=OrderType.STOP,
        stop_price=None,
        price=None,
    )
    result = order_manager.validator.validate(order)
    assert not result.ok


def test_trading_halted_instrument(execution_settings, kill_switch):
    meta = InstrumentMeta(symbol="AAPL", trading_enabled=False, reference_price=100.0)
    om = OrderManager(
        execution_settings,
        kill_switch=kill_switch,
        validator=OrderValidator(execution_settings, instruments={"AAPL": meta}),
    )
    order = om.create_order(
        instrument="AAPL", side=Side.BUY, quantity=10, order_type="LIMIT", price=100.0
    )
    assert not om.validator.validate(order).ok


def test_capital_notional_reject(order_manager):
    order = order_manager.create_order(
        instrument="AAPL",
        side=Side.BUY,
        quantity=100_000,
        order_type="LIMIT",
        price=100.0,  # 10M > max_order_notional 5M
    )
    result = order_manager.validator.validate(order)
    assert not result.ok
    assert any("notional" in e for e in result.errors)


def test_hard_risk_reject_blocks_approval(execution_settings, kill_switch):
    def risk_cb(order):
        return False, "hard risk reject"

    om = OrderManager(
        execution_settings,
        kill_switch=kill_switch,
        validate_risk=risk_cb,
    )
    order = om.create_order(
        instrument="AAPL",
        side=Side.BUY,
        quantity=10,
        order_type="LIMIT",
        price=100.0,
        urgency=Urgency.CRITICAL,
    )
    with pytest.raises(ValidationError):
        om.validate_and_approve(order.order_id)
    assert om.get(order.order_id).state is OrderState.REJECTED
    # CRITICAL urgency warning present
    result = om.validator.validate(order)
    assert any("CRITICAL" in w for w in result.warnings) or not result.ok


def test_hard_risk_reject_on_submit(execution_settings, kill_switch):
    """Risk callback may flip between validate and submit — urgency never overrides."""
    calls = {"n": 0}

    def risk_cb(order):
        calls["n"] += 1
        if calls["n"] == 1:
            return True, "ok"
        return False, "hard risk reject on submit"

    om = OrderManager(execution_settings, kill_switch=kill_switch, validate_risk=risk_cb)
    order = om.create_order(
        instrument="AAPL",
        side=Side.BUY,
        quantity=10,
        order_type="LIMIT",
        price=100.0,
        urgency=Urgency.CRITICAL,
    )
    om.validate_and_approve(order.order_id)
    with pytest.raises(ExecutionError) as ei:
        om.submit(order.order_id)
    assert ei.value.code == "HARD_RISK_REJECT"
    assert om.get(order.order_id).state is OrderState.REJECTED


def test_kill_switch_blocks_submit(order_manager, make_limit_order, kill_switch):
    order = make_limit_order(urgency=Urgency.CRITICAL)
    order_manager.validate_and_approve(order.order_id)
    kill_switch.engage_global("halted")
    with pytest.raises(ExecutionError) as ei:
        order_manager.submit(order.order_id)
    assert ei.value.code == "KILL_SWITCH_ACTIVE"
    assert order_manager.get(order.order_id).state is OrderState.FAILED


def test_kill_switch_account_venue_strategy(execution_settings):
    ks = KillSwitch()
    om = OrderManager(execution_settings, kill_switch=ks)
    order = om.create_order(
        instrument="AAPL",
        side=Side.BUY,
        quantity=10,
        order_type="LIMIT",
        price=100.0,
        account_id="ACC1",
        strategy_id="STR1",
        venue="SIM",
        urgency=Urgency.HIGH,
    )
    om.validate_and_approve(order.order_id)
    ks.engage_account("ACC1")
    with pytest.raises(ExecutionError) as ei:
        om.submit(order.order_id)
    assert ei.value.code == "KILL_SWITCH_ACTIVE"

    ks.clear_account("ACC1")
    ks.engage_venue("SIM")
    with pytest.raises(ExecutionError):
        om.submit(order.order_id)

    ks.clear_venue("SIM")
    ks.engage_strategy("STR1")
    with pytest.raises(ExecutionError):
        om.submit(order.order_id)


def test_require_risk_callback(execution_settings, kill_switch):
    settings = execution_settings.model_copy(
        update={"risk": RiskConfig(enforce_hard_limits=True, require_risk_callback=True)}
    )
    om = OrderManager(settings, kill_switch=kill_switch)
    order = om.create_order(
        instrument="AAPL", side=Side.BUY, quantity=10, order_type="LIMIT", price=100.0
    )
    result = om.validator.validate(order)
    assert not result.ok
    assert any("risk validation callback required" in e for e in result.errors)


def test_price_band_require_reference(execution_settings, kill_switch):
    settings = execution_settings.model_copy(
        update={"price_bands": PriceBandConfig(enabled=True, band_pct=0.05, require_reference=True)}
    )
    om = OrderManager(settings, kill_switch=kill_switch)
    order = om.create_order(
        instrument="XYZ", side=Side.BUY, quantity=10, order_type="LIMIT", price=100.0
    )
    result = om.validator.validate(order)
    assert not result.ok
    assert any("reference" in e for e in result.errors)


def test_validate_or_raise(order_manager):
    order = order_manager.create_order(
        instrument="AAPL", side=Side.BUY, quantity=10, order_type="LIMIT", price=100.0
    )
    ok = order_manager.validator.validate_or_raise(order)
    assert ok.ok
    bad = order_manager.create_order(
        instrument="AAPL", side=Side.BUY, quantity=0, order_type="MARKET"
    )
    with pytest.raises(ValidationError):
        order_manager.validator.validate_or_raise(bad)


def test_available_capital_check(execution_settings, kill_switch):
    validator = OrderValidator(execution_settings, available_capital=500.0)
    om = OrderManager(execution_settings, kill_switch=kill_switch, validator=validator)
    order = om.create_order(
        instrument="AAPL", side=Side.BUY, quantity=10, order_type="LIMIT", price=100.0
    )
    # notional 1000 > 500
    assert not om.validator.validate(order).ok
