# Event Engine

Deterministic event-driven backtesting: MARKET → SIGNAL → PORTFOLIO → ORDER → FILL → PnL. Priority queue + PIT clock.

---

## Purpose

The event engine is the causal core of Phase 13 backtests. It advances a timezone-aware clock, drains a deterministic priority queue, and dispatches typed handlers so that **no handler may observe information after `event.timestamp`**.

**Package:** `iqrp.app.backtesting.event_engine`  
**Primary type:** `EventDrivenEngine`  
**Related:** [BacktestingPlatform](BacktestingPlatform.md) · [Reproducibility](Reproducibility.md) · [Phase13](Phase13_BacktestingPlatform.md)

---

## Architecture

```text
BacktestClock (tz-aware, forward-only)
        │
EventScheduler ──seed/enqueue──► EventQueue (min-heap)
        │                              │
        └──────── EventDrivenEngine ◄──┘
                      │
         register(EventType → handlers)
                      │
              drain_at(timestamp) in priority order
                      │
              advance clock → dispatch handlers
```

Ordering key on the queue: `(timestamp, priority, sequence)`.

- Earlier timestamps first
- Within the same timestamp, **lower priority number first**
- Insertion sequence breaks remaining ties (full determinism)

---

## Event taxonomy and priorities

`EventType` / `EVENT_PRIORITY` (lower = earlier within the same timestamp):

| EventType | Priority |
|-----------|----------|
| `MARKET` | 10 |
| `FEATURE` | 20 |
| `SIGNAL` | 30 |
| `FORECAST` | 40 |
| `RISK` | 50 |
| `PORTFOLIO` | 60 |
| `ORDER` | 70 |
| `EXECUTION` | 80 |
| `FILL` | 90 |
| `POSITION` | 100 |
| `PNL` | 110 |
| `RISK_UPDATE` | 120 |
| `SETTLEMENT` | 130 |

Canonical flow:

```text
MARKET → FEATURE → SIGNAL → FORECAST → RISK → PORTFOLIO
  → ORDER → EXECUTION → FILL → POSITION → PNL → RISK_UPDATE → SETTLEMENT
```

Typed helpers: `MarketEvent`, `SignalEvent`, `ForecastEvent`, `RiskEvent` / `RiskUpdateEvent`, `PortfolioEvent`, `OrderEvent`, `FillEvent`, `SettlementEvent` — all build on base `Event`.

---

## Key classes

### `Event`

```python
from datetime import datetime, timezone
from iqrp.app.backtesting.event_engine import Event, EventType

ev = Event(
    timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
    event_type=EventType.MARKET,
    payload={"symbol": "XYZ", "close": 100.0},
)
```

- Timestamps **must** be timezone-aware (naive raises `ValueError`)
- `priority` defaults from `EVENT_PRIORITY`; override only when intentional
- `event_id` is a stable UUID hex for idempotency / audit
- `with_payload(**updates)` returns a merged copy

### `EventQueue`

```python
from iqrp.app.backtesting.event_engine import EventQueue

q = EventQueue()
q.put(ev)
nxt = q.peek()
batch = q.drain_at(ev.timestamp)   # all events at exact timestamp
upto = q.drain_until(asof)         # timestamp <= asof
```

### `BacktestClock`

```python
from datetime import datetime, timezone
from iqrp.app.backtesting.event_engine import BacktestClock, ClockFrequency

clock = BacktestClock(
    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
    frequency=ClockFrequency.DAILY,
    timezone="UTC",
)
clock.advance(1)           # +1 step
clock.advance_to(end)      # forward only — never jumps backwards
```

Frequencies: `tick`, `second`, `minute`, `hourly`, `daily`, `custom` (requires `step`).

### `EventScheduler`

Recurring jobs emit events on a fixed interval:

```python
from datetime import timedelta
from iqrp.app.backtesting.event_engine import EventScheduler, EventType

sched = EventScheduler()
sched.schedule_event_type(
    EventType.MARKET,
    interval=timedelta(days=1),
    start=start,
    end=end,
)
# EventDrivenEngine.run seeds jobs via seed_until / enqueue_due
```

`ScheduledJob` holds `job_id`, `interval`, `factory`, `next_time`, optional `end`.

### `EventDrivenEngine`

```python
from iqrp.app.backtesting.event_engine import EventDrivenEngine, EventType

engine = EventDrivenEngine(clock=clock, on_invalidate=lambda reason: print(reason))

def on_market(event):
    # CRITICAL: only data with effective time <= event.timestamp
    ...

engine.register(EventType.MARKET, on_market)
engine.register(None, audit_all)   # wildcard after type-specific handlers
engine.submit(ev)
state = engine.run(start=start, end=end, advance_empty_ticks=False)
```

Run algorithm:

1. Optionally reset/advance clock to `start`
2. Seed due scheduled jobs through `end`
3. While next event `timestamp <= end`: advance clock, enqueue due jobs, drain same-timestamp batch in priority order
4. Optionally advance empty ticks when idle
5. Advance clock to `end` → `COMPLETED` (unless `INVALIDATED` / `FAILED`)

Out-of-order events (timestamp &lt; clock) raise `LookaheadError`. Call `invalidate(reason)` to mark the backtest invalid (e.g. after PIT violation).

---

## Critical rules

| Rule | Detail |
|------|--------|
| PIT boundary | Before every handler, clock advances to `event.timestamp`. Handlers must not read data with effective time strictly after that. |
| Enforce via PIT helpers | Use `assert_no_lookahead`, `filter_universe_asof`, `detect_leakage`, `actions_asof` at data boundaries. |
| Invalidate on violation | Look-ahead → `BacktestState.INVALIDATED`, not silent continue. |
| Determinism | Same seed, same event stream, same insertion order → identical processing order. |
| Aware timestamps only | Naive datetimes are rejected on events and corporate actions. |
| Forward-only clock | `BacktestClock` never moves backwards. |

```python
from iqrp.app.backtesting.pit import assert_no_lookahead

def on_signal(event):
    assert_no_lookahead(feature_asof, event.timestamp, context="signal_feature")
```

---

## Integration

- Used by the institutional platform for event-driven experiments; `BacktestEngine.run` also provides a bar-loop simulator with the same PIT contract.
- Execution / Risk / Portfolio participate as **handlers or imported estimators**, not as mutated packages.
- Optional Execution TCA remains import-only when costing fills.

---

## Example: minimal MARKET → SIGNAL → PNL loop

```python
from datetime import datetime, timedelta, timezone
from iqrp.app.backtesting.event_engine import (
    BacktestClock,
    Event,
    EventDrivenEngine,
    EventType,
)

start = datetime(2024, 1, 1, tzinfo=timezone.utc)
clock = BacktestClock(start=start, frequency="daily")
eng = EventDrivenEngine(clock=clock)

pnl = {"last_signal": 0.0, "cum": 0.0}

def on_market(e):
    ret = float(e.payload["ret"])
    pnl["cum"] += pnl["last_signal"] * ret

def on_signal(e):
    pnl["last_signal"] = float(e.payload["weight"])

eng.register(EventType.MARKET, on_market)
eng.register(EventType.SIGNAL, on_signal)

t = start
for i, r in enumerate([0.01, -0.005, 0.002]):
    t = start + timedelta(days=i)
    # Same timestamp: MARKET (10) before SIGNAL (30)
    eng.submit(Event(t, EventType.MARKET, payload={"ret": r}))
    eng.submit(Event(t, EventType.SIGNAL, payload={"weight": 1.0 if r > 0 else -1.0}))

eng.run(start=start, end=start + timedelta(days=3))
```
