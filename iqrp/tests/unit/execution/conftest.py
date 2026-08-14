"""Shared fixtures for Institutional Execution Platform unit tests."""

from __future__ import annotations

from typing import Any

import pytest

from iqrp.app.execution import (
    ExecutionEngine,
    ExecutionSettings,
    KillSwitch,
    OrderManager,
    SimulatedVenue,
)
from iqrp.app.execution.config import (
    CapitalConfig,
    FillConfig,
    KillSwitchConfig,
    PriceBandConfig,
    RiskConfig,
    TickLotConfig,
)
from iqrp.app.execution.order_manager.order_validator import InstrumentMeta, OrderValidator
from iqrp.app.execution.smart_routing import SmartRouter
from iqrp.app.execution.types import Side, Urgency

SEED = 42


@pytest.fixture
def seed() -> int:
    return SEED


@pytest.fixture
def execution_settings() -> ExecutionSettings:
    return ExecutionSettings(
        seed=SEED,
        default_venue="SIM",
        default_urgency="NORMAL",
        tick_lot=TickLotConfig(
            default_tick_size=0.01,
            default_lot_size=1.0,
            min_qty=1.0,
            max_qty=1_000_000.0,
        ),
        price_bands=PriceBandConfig(enabled=True, band_pct=0.10, require_reference=False),
        capital=CapitalConfig(
            check_enabled=True,
            max_notional=10_000_000.0,
            max_order_notional=5_000_000.0,
        ),
        risk=RiskConfig(enforce_hard_limits=True, require_risk_callback=False),
        kill_switch=KillSwitchConfig(check_on_submit=True),
        fills=FillConfig(idempotent=True, allow_overfill=False),
    )


@pytest.fixture
def kill_switch() -> KillSwitch:
    return KillSwitch()


@pytest.fixture
def market_context() -> dict[str, Any]:
    """Flat market context for AAPL — fixed seed values, small simulation."""
    return {
        "mid": 100.0,
        "price": 100.0,
        "spread": 0.02,
        "adv": 1_000_000.0,
        "volatility": 0.02,
        "vwap": 100.0,
        "twap": 100.0,
        "n_slices": 3,
        "horizon_seconds": 60.0,
    }


@pytest.fixture
def nested_market_context(market_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "AAPL": dict(market_context),
        "MSFT": {**market_context, "mid": 250.0, "price": 250.0},
    }


@pytest.fixture
def instrument_meta() -> InstrumentMeta:
    return InstrumentMeta(
        symbol="AAPL",
        tick_size=0.01,
        lot_size=1.0,
        min_qty=1.0,
        max_qty=100_000.0,
        trading_enabled=True,
        reference_price=100.0,
    )


@pytest.fixture
def approving_risk() -> Any:
    class ApprovingRisk:
        def validate_position(self, *args: Any, **kwargs: Any) -> tuple[bool, str]:
            return True, "ok"

        def check_limits(self, *args: Any, **kwargs: Any) -> list[Any]:
            return []

    return ApprovingRisk()


@pytest.fixture
def rejecting_risk() -> Any:
    class RejectingRisk:
        def validate_position(self, *args: Any, **kwargs: Any) -> Any:
            class _D:
                approved = False
                reason = "hard risk reject"

            return _D()

        def check_limits(self, *args: Any, **kwargs: Any) -> list[Any]:
            return [{"name": "max_position", "severity": "hard"}]

    return RejectingRisk()


@pytest.fixture
def order_manager(execution_settings: ExecutionSettings, kill_switch: KillSwitch) -> OrderManager:
    return OrderManager(execution_settings, kill_switch=kill_switch)


@pytest.fixture
def order_manager_with_meta(
    execution_settings: ExecutionSettings,
    kill_switch: KillSwitch,
    instrument_meta: InstrumentMeta,
) -> OrderManager:
    validator = OrderValidator(
        execution_settings,
        instruments={"AAPL": instrument_meta},
    )
    return OrderManager(
        execution_settings,
        kill_switch=kill_switch,
        validator=validator,
    )


@pytest.fixture
def simulated_venue(market_context: dict[str, Any]) -> SimulatedVenue:
    return SimulatedVenue(
        venue_id="SIM",
        instruments={"AAPL", "MSFT"},
        mode="fill",
        mid=float(market_context["mid"]),
        spread=float(market_context["spread"]),
        adv=float(market_context["adv"]),
        available_qty=1e9,
    )


@pytest.fixture
def venue_fill(simulated_venue: SimulatedVenue) -> SimulatedVenue:
    return simulated_venue


@pytest.fixture
def venue_partial(market_context: dict[str, Any]) -> SimulatedVenue:
    return SimulatedVenue(
        venue_id="SIM_PARTIAL",
        instruments={"AAPL"},
        mode="partial",
        partial_fraction=0.5,
        mid=float(market_context["mid"]),
        spread=float(market_context["spread"]),
        adv=float(market_context["adv"]),
    )


@pytest.fixture
def venue_reject(market_context: dict[str, Any]) -> SimulatedVenue:
    return SimulatedVenue(
        venue_id="SIM_REJECT",
        instruments={"AAPL"},
        mode="reject",
        reject_reason="simulated_reject",
        mid=float(market_context["mid"]),
        spread=float(market_context["spread"]),
    )


@pytest.fixture
def venue_ack(market_context: dict[str, Any]) -> SimulatedVenue:
    return SimulatedVenue(
        venue_id="SIM_ACK",
        instruments={"AAPL"},
        mode="ack",
        mid=float(market_context["mid"]),
        spread=float(market_context["spread"]),
    )


@pytest.fixture
def multi_venues(market_context: dict[str, Any]) -> list[SimulatedVenue]:
    mid = float(market_context["mid"])
    spread = float(market_context["spread"])
    return [
        SimulatedVenue(
            venue_id="SIM_A",
            instruments={"AAPL"},
            mode="fill",
            mid=mid,
            spread=spread,
            adv=2e6,
            available_qty=5e5,
        ),
        SimulatedVenue(
            venue_id="SIM_B",
            instruments={"AAPL"},
            mode="fill",
            mid=mid + 0.01,
            spread=spread * 1.2,
            adv=1e6,
            available_qty=3e5,
        ),
    ]


@pytest.fixture
def smart_router(kill_switch: KillSwitch) -> SmartRouter:
    return SmartRouter(kill_switch=kill_switch, mode="single")


@pytest.fixture
def engine(
    execution_settings: ExecutionSettings,
    kill_switch: KillSwitch,
) -> ExecutionEngine:
    return ExecutionEngine(settings=execution_settings, kill_switch=kill_switch)


@pytest.fixture
def engine_with_risk(
    execution_settings: ExecutionSettings,
    kill_switch: KillSwitch,
    approving_risk: Any,
) -> ExecutionEngine:
    return ExecutionEngine(
        settings=execution_settings,
        kill_switch=kill_switch,
        risk_engine=approving_risk,
    )


@pytest.fixture
def make_limit_order(order_manager: OrderManager):
    def _make(
        *,
        instrument: str = "AAPL",
        side: Side | str = Side.BUY,
        quantity: float = 100.0,
        price: float = 100.0,
        urgency: Urgency | str = Urgency.NORMAL,
        account_id: str | None = None,
        strategy_id: str | None = None,
        venue: str | None = "SIM",
        idempotency_key: str | None = None,
    ):
        return order_manager.create_order(
            instrument=instrument,
            side=side,
            quantity=quantity,
            order_type="LIMIT",
            price=price,
            urgency=urgency,
            account_id=account_id,
            strategy_id=strategy_id,
            venue=venue,
            idempotency_key=idempotency_key,
        )

    return _make
