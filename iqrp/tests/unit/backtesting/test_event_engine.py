"""Event engine: queue ordering, clock frequencies, scheduler, typed events."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from iqrp.app.backtesting.event_engine import (
    EVENT_PRIORITY,
    BacktestClock,
    ClockFrequency,
    Event,
    EventDrivenEngine,
    EventQueue,
    EventScheduler,
    EventType,
    FillEvent,
    ForecastEvent,
    LookaheadError,
    MarketEvent,
    OrderEvent,
    PortfolioEvent,
    RiskEvent,
    RiskUpdateEvent,
    SettlementEvent,
    SignalEvent,
    priority_for,
)
from iqrp.app.backtesting.types import BacktestState


def _ts(day: int = 1, hour: int = 0) -> datetime:
    return datetime(2020, 1, day, hour, tzinfo=UTC)


# --------------------------------------------------------------------------- queue
def test_event_queue_priority_order() -> None:
    q = EventQueue()
    ts = _ts()
    # Insert out of priority order
    for et in (
        EventType.FILL,
        EventType.MARKET,
        EventType.SIGNAL,
        EventType.ORDER,
        EventType.PORTFOLIO,
    ):
        q.put(Event(ts, et))
    order = [q.get().event_type for _ in range(5)]
    assert order == [
        EventType.MARKET,
        EventType.SIGNAL,
        EventType.PORTFOLIO,
        EventType.ORDER,
        EventType.FILL,
    ]


def test_event_queue_timestamp_then_priority_then_sequence() -> None:
    q = EventQueue()
    t0, t1 = _ts(1), _ts(2)
    q.put(Event(t1, EventType.MARKET))
    q.put(Event(t0, EventType.FILL))
    q.put(Event(t0, EventType.MARKET))
    q.put(Event(t0, EventType.MARKET, payload={"seq": 2}))  # later sequence
    first = q.get()
    assert first.timestamp == t0 and first.event_type == EventType.MARKET
    second = q.get()
    assert second.event_type == EventType.MARKET
    third = q.get()
    assert third.event_type == EventType.FILL
    fourth = q.get()
    assert fourth.timestamp == t1


def test_event_queue_peek_drain_clear() -> None:
    q = EventQueue()
    assert q.empty()
    assert q.peek() is None
    with pytest.raises(IndexError):
        q.get()
    t0, t1 = _ts(1), _ts(2)
    q.put(Event(t0, EventType.MARKET))
    q.put(Event(t0, EventType.SIGNAL))
    q.put(Event(t1, EventType.MARKET))
    assert len(q) == 3
    assert q.peek().event_type == EventType.MARKET
    batch = q.drain_at(t0)
    assert [e.event_type for e in batch] == [EventType.MARKET, EventType.SIGNAL]
    until = q.drain_until(t1)
    assert len(until) == 1
    q.put(Event(t0, EventType.PNL))
    q.clear()
    assert q.empty()
    with pytest.raises(TypeError):
        q.put("not-an-event")  # type: ignore[arg-type]


def test_event_queue_iter() -> None:
    q = EventQueue()
    ts = _ts()
    q.put(Event(ts, EventType.MARKET))
    q.put(Event(ts, EventType.SIGNAL))
    types = [e.event_type for e in q]
    assert types == [EventType.MARKET, EventType.SIGNAL]


# --------------------------------------------------------------------------- clock
@pytest.mark.parametrize(
    "freq,delta",
    [
        (ClockFrequency.TICK, timedelta(microseconds=1)),
        ("second", timedelta(seconds=1)),
        ("minute", timedelta(minutes=1)),
        ("hourly", timedelta(hours=1)),
        ("daily", timedelta(days=1)),
        ("custom", timedelta(seconds=30)),
    ],
)
def test_backtest_clock_frequencies(freq, delta) -> None:
    start = _ts()
    kwargs = {"frequency": freq}
    if str(freq) == "custom" or freq == ClockFrequency.CUSTOM:
        kwargs["step"] = delta
        kwargs["frequency"] = "custom"
    clock = BacktestClock(start, **kwargs)
    assert clock.now == start
    clock.tick()
    assert clock.now == start + clock.step
    assert clock.current == clock.now
    assert clock.start == start


def test_backtest_clock_aliases_and_errors() -> None:
    start = _ts()
    for alias in ("sec", "min", "h", "day", "days"):
        c = BacktestClock(start, frequency=alias)
        assert c.step > timedelta(0)
    with pytest.raises(ValueError):
        BacktestClock(start, frequency="custom")  # missing step
    with pytest.raises(ValueError):
        BacktestClock(start, frequency="bogus")
    with pytest.raises(ValueError):
        BacktestClock(start, frequency="custom", step=timedelta(0))
    with pytest.raises(ValueError):
        BacktestClock(start, frequency="custom", step=timedelta(seconds=-1))
    clock = BacktestClock(start, frequency="daily")
    with pytest.raises(ValueError):
        clock.set(start - timedelta(days=1))
    with pytest.raises(ValueError):
        clock.advance(-1)
    assert clock.advance(0) == clock.now
    clock.reset()
    assert clock.now == start
    clock.advance_to(start + timedelta(days=3))
    assert clock.now == start + timedelta(days=3)
    aware = clock.ensure_aware(datetime(2020, 2, 1))
    assert aware.tzinfo is not None


def test_backtest_clock_range_and_tz() -> None:
    start = _ts()
    clock = BacktestClock(start, frequency="daily", timezone="UTC")
    times = list(clock.range(start + timedelta(days=2), inclusive=True))
    assert len(times) >= 2
    clock2 = BacktestClock(datetime(2020, 1, 1), frequency="daily")  # naive → aware
    assert clock2.now.tzinfo is not None
    clock3 = BacktestClock(start, frequency="daily", timezone="America/New_York")
    assert clock3.tzinfo is not None
    # iterator yields then advances
    it = iter(BacktestClock(start, frequency="daily"))
    t0 = next(it)
    t1 = next(it)
    assert t1 == t0 + timedelta(days=1)


# --------------------------------------------------------------------------- events
def test_event_priority_canonical_order() -> None:
    expected = [
        EventType.MARKET,
        EventType.FEATURE,
        EventType.SIGNAL,
        EventType.FORECAST,
        EventType.RISK,
        EventType.PORTFOLIO,
        EventType.ORDER,
        EventType.EXECUTION,
        EventType.FILL,
        EventType.POSITION,
        EventType.PNL,
        EventType.RISK_UPDATE,
        EventType.SETTLEMENT,
    ]
    prios = [priority_for(et) for et in expected]
    assert prios == sorted(prios)
    assert EVENT_PRIORITY[EventType.MARKET] < EVENT_PRIORITY[EventType.SIGNAL]


def test_event_requires_aware_timestamp() -> None:
    with pytest.raises(ValueError):
        Event(datetime(2020, 1, 1), EventType.MARKET)
    e = Event(_ts(), "MARKET", payload={"x": 1})
    e2 = e.with_payload(y=2)
    assert e2.payload["x"] == 1 and e2.payload["y"] == 2
    assert "MARKET" in repr(e)


def test_typed_events() -> None:
    ts = _ts()
    m = MarketEvent(ts, {"symbol": "AAPL", "close": 150.0})
    assert m.symbol == "AAPL" and m.close == 150.0
    assert SignalEvent(ts, {"strength": 0.5}).event_type == EventType.SIGNAL
    assert ForecastEvent(ts, {"mu": 0.01}).event_type == EventType.FORECAST
    assert RiskEvent(ts, {"var": 0.02}).event_type == EventType.RISK
    assert RiskUpdateEvent(ts, {"var": 0.03}).event_type == EventType.RISK_UPDATE
    assert PortfolioEvent(ts, {"w": 1.0}).event_type == EventType.PORTFOLIO
    assert OrderEvent(ts, {"qty": 10}).event_type == EventType.ORDER
    assert FillEvent(ts, {"qty": 10, "price": 100}).event_type == EventType.FILL
    assert SettlementEvent(ts, {"cash": 1.0}).event_type == EventType.SETTLEMENT


def test_full_pipeline_event_order() -> None:
    """Market→Features→Signals→Forecasts→Risk→Portfolio→Orders→Execution→Fills→Positions→PnL→RiskUpdate."""
    q = EventQueue()
    ts = _ts()
    pipeline = [
        EventType.MARKET,
        EventType.FEATURE,
        EventType.SIGNAL,
        EventType.FORECAST,
        EventType.RISK,
        EventType.PORTFOLIO,
        EventType.ORDER,
        EventType.EXECUTION,
        EventType.FILL,
        EventType.POSITION,
        EventType.PNL,
        EventType.RISK_UPDATE,
    ]
    # reverse insert
    for et in reversed(pipeline):
        q.put(Event(ts, et))
    got = [q.get().event_type for _ in pipeline]
    assert got == pipeline


# --------------------------------------------------------------------------- scheduler
def test_event_scheduler_seed_and_cancel() -> None:
    sched = EventScheduler()
    start, end = _ts(1), _ts(5)
    jid = sched.schedule_event_type(
        EventType.MARKET, interval=timedelta(days=1), start=start, end=end
    )
    q = EventQueue()
    clock = BacktestClock(start, frequency="daily")
    n = sched.seed_until(q, start=start, end=end, clock=clock)
    assert n >= 4
    assert sched.cancel(jid) is True
    assert sched.cancel("missing") is False
    assert sched.remove(jid) is True
    assert len(sched.jobs()) == 0
    with pytest.raises(ValueError):
        sched.schedule(interval=timedelta(0), factory=lambda t: Event(t, EventType.MARKET), start=start)
    with pytest.raises(ValueError):
        sched.schedule(
            interval=timedelta(days=1),
            factory=lambda t: Event(t, EventType.MARKET),
            start=datetime(2020, 1, 1),
        )


def test_scheduler_enqueue_due() -> None:
    sched = EventScheduler()
    start = _ts(1)
    sched.schedule_event_type(EventType.SIGNAL, interval=timedelta(days=1), start=start)
    q = EventQueue()
    emitted = sched.enqueue_due(q, start + timedelta(days=2))
    assert len(emitted) >= 3


# --------------------------------------------------------------------------- engine
def test_event_driven_engine_run() -> None:
    start, end = _ts(1), _ts(3)
    clock = BacktestClock(start, frequency="daily")
    eng = EventDrivenEngine(clock=clock)
    seen: list[EventType] = []
    eng.register(EventType.MARKET, lambda e: seen.append(e.event_type))
    eng.register(None, lambda e: None)
    eng.submit(MarketEvent(start, {"close": 1}))
    eng.submit(SignalEvent(start, {}))
    eng.submit(MarketEvent(start + timedelta(days=1), {"close": 2}))
    state = eng.run(start=start, end=end)
    assert state == BacktestState.COMPLETED
    assert eng.processed_count >= 2
    assert EventType.MARKET in seen
    assert eng.unregister(EventType.MARKET, eng._handlers[EventType.MARKET][0] if False else (lambda e: None)) is False


def test_event_driven_engine_invalidate_and_lookahead() -> None:
    reasons: list[str] = []
    start = _ts(1)
    clock = BacktestClock(start, frequency="daily")
    eng = EventDrivenEngine(clock=clock, on_invalidate=reasons.append)
    eng.invalidate("leak")
    assert eng.state == BacktestState.INVALIDATED
    assert reasons == ["leak"]
    assert eng.run(end=_ts(2)) == BacktestState.INVALIDATED

    clock2 = BacktestClock(start + timedelta(days=2), frequency="daily")
    eng2 = EventDrivenEngine(clock=clock2)
    eng2.submit(MarketEvent(start, {}))  # past event
    with pytest.raises(LookaheadError):
        eng2.run(end=start + timedelta(days=3))
    assert eng2.state == BacktestState.FAILED


def test_event_driven_engine_requires_end_and_max_events() -> None:
    clock = BacktestClock(_ts(), frequency="daily")
    eng = EventDrivenEngine(clock=clock)
    with pytest.raises(ValueError):
        eng.run()
    eng.submit(MarketEvent(_ts(), {}))
    eng.submit(MarketEvent(_ts(2), {}))
    eng.run(end=_ts(5), max_events=1)
    assert eng.processed_count == 1


def test_event_driven_engine_advance_empty_ticks() -> None:
    start = _ts(1)
    clock = BacktestClock(start, frequency="daily")
    eng = EventDrivenEngine(clock=clock)
    sched = eng.scheduler
    sched.schedule_event_type(
        EventType.MARKET, interval=timedelta(days=1), start=start + timedelta(days=1), end=_ts(3)
    )
    # Empty queue initially — advance_empty_ticks seeds via seed_until actually
    # but also idle path
    state = eng.run(start=start, end=_ts(3), advance_empty_ticks=True)
    assert state == BacktestState.COMPLETED
