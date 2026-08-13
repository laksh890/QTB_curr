"""Deterministic event-driven backtesting engine.

Event priority (lower = earlier within the same timestamp)::

    MARKET=10 → FEATURE=20 → SIGNAL=30 → FORECAST=40 → RISK=50 →
    PORTFOLIO=60 → ORDER=70 → EXECUTION=80 → FILL=90 → POSITION=100 →
    PNL=110 → RISK_UPDATE=120 → SETTLEMENT=130

CRITICAL: No handler may access data after ``event.timestamp``.
Use :mod:`iqrp.app.backtesting.pit` to enforce the boundary.
"""

from __future__ import annotations

from iqrp.app.backtesting.event_engine.clock import BacktestClock, ClockFrequency
from iqrp.app.backtesting.event_engine.engine import (
    EventDrivenEngine,
    EventHandler,
    LookaheadError,
)
from iqrp.app.backtesting.event_engine.event import (
    EVENT_PRIORITY,
    Event,
    EventType,
    priority_for,
)
from iqrp.app.backtesting.event_engine.event_queue import EventQueue
from iqrp.app.backtesting.event_engine.fill_event import FillEvent
from iqrp.app.backtesting.event_engine.forecast_event import ForecastEvent
from iqrp.app.backtesting.event_engine.market_event import MarketEvent
from iqrp.app.backtesting.event_engine.order_event import OrderEvent
from iqrp.app.backtesting.event_engine.portfolio_event import PortfolioEvent
from iqrp.app.backtesting.event_engine.risk_event import RiskEvent, RiskUpdateEvent
from iqrp.app.backtesting.event_engine.scheduler import (
    EventFactory,
    EventScheduler,
    ScheduledJob,
)
from iqrp.app.backtesting.event_engine.settlement_event import SettlementEvent
from iqrp.app.backtesting.event_engine.signal_event import SignalEvent

__all__ = [
    "EVENT_PRIORITY",
    "BacktestClock",
    "ClockFrequency",
    "Event",
    "EventDrivenEngine",
    "EventFactory",
    "EventHandler",
    "EventQueue",
    "EventScheduler",
    "EventType",
    "FillEvent",
    "ForecastEvent",
    "LookaheadError",
    "MarketEvent",
    "OrderEvent",
    "PortfolioEvent",
    "RiskEvent",
    "RiskUpdateEvent",
    "ScheduledJob",
    "SettlementEvent",
    "SignalEvent",
    "priority_for",
]
