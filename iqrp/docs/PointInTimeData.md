# Point-in-Time Data

Point-in-time (PIT) helpers for historical features, signals, universes, and operational runner enforcement.

---

## Purpose

Ensure no strategy, feature, or universe membership uses information strictly after the simulation clock / event timestamp. Look-ahead and leakage invalidate the backtest.

**Data helpers:** `iqrp.app.backtesting.data.point_in_time`  
**Core primitives:** `iqrp.app.backtesting.pit`  
**Runner enforcement:** `EventPipeline._pit_check` when `BacktestRunConfig.enforce_pit=True` (default)  
**Related:** [Reproducibility](Reproducibility.md) · [DataPipeline](DataPipeline.md) · [BacktestExecution](BacktestExecution.md) · [EventEngine](EventEngine.md)

---

## Architecture

```text
Bar / feature / signal row
   timestamp (+ optional effective_timestamp)
            │
   assert_no_lookahead(data_ts, event_ts)
            │
   filter_*_asof(asof)  → only rows with effective ≤ asof
            │
   UniverseSpec.asof(when) / filter_universe_membership_asof
            │
   Runner: LookaheadViolation → context.invalidated → INVALIDATED
```

Naive datetimes are rejected (`LookaheadViolation`).

---

## Effective timestamps

```python
from iqrp.app.backtesting.data.point_in_time import (
    effective_timestamp,
    ensure_effective_timestamps,
    LookaheadViolation,
)

df = ensure_effective_timestamps(frame)
# effective_timestamp defaults to timestamp
# raises if effective_timestamp > timestamp (future leak into past label)
```

Prefer `effective_timestamp` when the observation becomes knowable after the bar’s calendar stamp (e.g. delayed fundamentals). For pure OHLCV bars, `timestamp == effective_timestamp` is typical.

---

## Filtering APIs

| Function | Role |
|----------|------|
| `filter_frame_asof_df(frame, asof)` | Rows with effective ≤ asof |
| `filter_features_asof` | Feature panels |
| `filter_signals_asof` | Signal / forecast panels |
| `filter_universe_membership_asof` | Membership windows → instrument list |
| `assert_no_lookahead(data_ts, event_ts, context=...)` | Hard check |
| `available_asof` / `filter_frame_asof` / `filter_universe_asof` | Re-exported from `pit` |

```python
from datetime import datetime, timezone
from iqrp.app.backtesting.data.point_in_time import filter_features_asof

asof = datetime(2020, 6, 1, tzinfo=timezone.utc)
safe = filter_features_asof(features, asof)
```

---

## Universes (survivorship)

`UniverseSpec` kinds: `single`, `list`, `historical`, `index_constituents`, `futures`, `continuous`, `custom`.

Historical membership rows use `instrument`/`symbol`, `start`, optional `end`. Resolution at `asof` excludes not-yet-listed and already-delisted names:

```python
from iqrp.app.backtesting.data import historical_universe

univ = historical_universe(
    [
        {"instrument": "AAA", "start": "2018-01-01T00:00:00+00:00", "end": "2021-01-01T00:00:00+00:00"},
        {"instrument": "BBB", "start": "2019-06-01T00:00:00+00:00"},
    ],
    name="demo",
)
names = univ.asof(datetime(2020, 1, 1, tzinfo=timezone.utc))
```

Static `universe: [...]` on `BacktestRunConfig` filters the loaded frame and portfolio targets; for true index reconstitutions, supply membership windows and resolve per date in strategy/research code.

---

## Corporate actions and continuous contracts

- `corporate_actions_asof` / platform `actions_asof`: only actions with `ex_date <= asof`.
- Continuous futures distinguish **raw**, **continuous research** (stitched/adjusted), and **tradable** front-month series (`ContractSeriesKind`). Research series must not be treated as live tradable without roll awareness.

---

## Runner enforcement

On each MARKET bar, `EventPipeline` compares bar timestamp to event timestamp via `assert_no_lookahead`. Violations set `context.invalidated`, call `engine.invalidate`, and the runner transitions to `INVALIDATED`.

Disable only for controlled diagnostics: `enforce_pit: false` on the run config (not recommended for institutional evidence).

---

## Critical rules

| Rule | Detail |
|------|--------|
| Tz-aware only | Naive timestamps raise |
| Data ≤ event time | Strictly after → invalidate |
| Survivorship | Membership as-of, not full-sample union |
| Future corporate actions forbidden | Filter with as-of |
| Leakage is terminal | Do not patch and continue |
