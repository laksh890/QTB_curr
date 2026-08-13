# Execution Platform

Institutional execution for IQRP: convert **approved** portfolio targets into parent/child orders, slice with algorithms, route across venues, apply fills idempotently, and produce pre/post-trade TCA — without generating alpha or inventing positions.

**Package:** `iqrp.app.execution`  
**Primary type:** `ExecutionEngine`  
**Hydra config:** `iqrp/configs/execution/default.yaml`

Related: [OrderManager](OrderManager.md) · [ExecutionAlgorithms](ExecutionAlgorithms.md) · [Slippage](Slippage.md) · [ExecutionCosts](ExecutionCosts.md) · [SmartRouting](SmartRouting.md) · [OrderLifecycle](OrderLifecycle.md) · [ExecutionRisk](ExecutionRisk.md) · [PositionReconciliation](PositionReconciliation.md) · [Phase 12 summary](Phase12_ExecutionPlatform.md)

> **Phase numbering:** Product briefs sometimes call this “Phase 11 Execution.” In this repository, Phase 11 is [Alpha Research](Phase11_AlphaResearch.md). The Institutional Execution Platform is **Phase 12**.

---

## Placement

```text
Alpha Research (signals) ──► Portfolio Construction (targets / weights)
                                      │
                                      ▼
                         approved current → target deltas
                                      │
           Risk Intelligence ◄────────┤  (authoritative when risk_engine provided)
                                      │
                                      ▼
                            ExecutionEngine
         plan → validate → estimate → algo slice → route → submit → fill
                                      │
                                      ▼
              fills · TCA · analytics · reconciliation · audit
```

**Portfolio vs Execution boundary**

| Concern | Portfolio (`iqrp.app.portfolio`) | Execution (`iqrp.app.execution`) |
|---------|----------------------------------|----------------------------------|
| Alpha / forecasts | Expresses caller-supplied forecasts under constraints | Never generates alpha |
| Positions | Produces target weights / positions | Converts **approved** current→target deltas only |
| Transaction costs | Model TC inside optimization ([TransactionCosts](TransactionCosts.md)) | Pre/post-trade TCA ([ExecutionCosts](ExecutionCosts.md)) |
| Risk | Pre-trade risk gate on construction | Hard risk + kill switches before every submit |
| Urgency | N/A | Scales slice aggressiveness; never overrides hard risk |

---

## Architectural rules (14)

| # | Rule |
|---|------|
| 1 | **No alpha generation.** Execution never invents signals, forecasts, or positions. |
| 2 | **Approved residual only.** Child plans and fills must never exceed the approved parent residual / target delta. |
| 3 | **Risk Intelligence is authoritative** when `risk_engine` (or `validate_risk`) is provided — reject cannot be bypassed. |
| 4 | **Kill switches are fail-safe.** Global / account / venue / strategy halt blocks new submits and routing. |
| 5 | **Urgency never overrides hard risk or kill switches.** It only changes slice count, POV rate, and limit aggression. |
| 6 | **Idempotent events and fills.** Duplicate `event_id` / `idempotency_key` values are no-ops. |
| 7 | **On HALT:** stop new orders; optionally cancel open working orders (`cancel_open=True`). |
| 8 | **Point-in-time only.** No future prices, curves, or fills may enter planning or TCA. |
| 9 | **Hard limits never silently relaxed.** Tick/lot, bands, capital, participation, and risk rejects fail closed. |
| 10 | **Portfolio owns targets; Execution owns microstructure.** Construction decides *what*; execution decides *how* within residual. |
| 11 | **Illegal state transitions rejected.** Order and execution state machines never coerce invalid edges. |
| 12 | **No silent overfills.** Overfill rejected unless `fills.allow_overfill` is explicitly true. |
| 13 | **Routing safety.** Never route to unavailable / halted / unsupported / kill-blocked / risk-rejected venues. |
| 14 | **Auditable and reproducible.** Decisions carry `seed`, `data_version`, `model_version`, and append-only audit. |

---

## Package architecture

```text
iqrp.app.execution
├── engine.py              ExecutionEngine orchestrator
├── types.py               Side, Urgency, OrderType, TimeInForce, KillSwitch
├── config.py              ExecutionSettings (Hydra)
├── registry.py            algorithm name → planner
├── order_manager/         Order, ParentOrder, lifecycle, fills, recon
├── algorithms/            TWAP, VWAP, POV, IS, Adaptive, …
├── slippage/              expected + realized slippage / impact
├── transaction_costs/     pre-trade estimate + post-trade TCA
├── smart_routing/         venues, scoring, allocation, fallback
├── analytics.py           execution quality report
├── latency.py             LatencyTracker
└── simulation.py          historical / synthetic fill paths
```

---

## Quick start

```python
from iqrp.app.execution import (
    ExecutionEngine,
    ExecutionSettings,
    KillSwitch,
    Urgency,
)

engine = ExecutionEngine(ExecutionSettings.default())

market = {
    "AAPL": {"mid": 190.0, "spread": 0.02, "adv": 5e7, "volatility": 0.02},
}

report = engine.execute(
    {"AAPL": 1000.0},           # target positions
    current={"AAPL": 0.0},
    algo="twap",
    urgency=Urgency.NORMAL,
    market_context=market,
    strategy_id="strat_demo",
)

print(report.status, report.state, len(report.fills))
print(report.pre_trade["by_parent"][0]["costs"]["total_cost_bps"])
```

Kill-switch example (fail-safe):

```python
ks = KillSwitch()
ks.engage_global("ops drill")
engine = ExecutionEngine(kill_switch=ks)
# engine.execute(...)  → ExecutionError(code="KILL_SWITCH_ACTIVE")
```

---

## `ExecutionEngine` API

### Construction

```python
ExecutionEngine(
    settings: ExecutionSettings | None = None,
    order_manager: OrderManager | None = None,
    router: SmartRouter | None = None,
    kill_switch: KillSwitch | None = None,
    risk_engine: Any | None = None,   # Risk Intelligence adapter
)
```

When `risk_engine` is set, the engine wires `validate_position` / `check_limits` into `OrderManager` as a hard gate.

### Planning and estimation

| Method | Purpose |
|--------|---------|
| `plan_from_targets(current, target, **kwargs)` | `target_to_orders` → `Order` list (no alpha) |
| `estimate_costs(orders_or_delta, market_context)` | Pre-trade TCA via `pre_trade_cost_estimate` |
| `estimate_slippage(...)` | Pre-trade expected slippage breakdown |

### Execute

```python
report: ExecutionReport = engine.execute(
    parent_order_or_targets,   # ParentOrder | Order | {inst: qty} | [Order]
    *,
    algo: str = "twap",
    urgency: Urgency | str = Urgency.NORMAL,
    venues=None,               # defaults to SimulatedVenue(settings.default_venue)
    market_context=None,
    current=None,              # required context when targets are absolute
    simulation_mode=None,      # auto True when venues are SimulatedVenue / None
    account_id=None,
    strategy_id=None,
)
```

**Flow:** `PLANNING` → `VALIDATING` → algo `plan` → residual clip → route → validate/approve → submit → (sim) ack/fill → post-trade TCA → analytics → `COMPLETED` / `PARTIALLY_EXECUTED` / `FAILED` / raise on kill.

### Routing, events, reconciliation

| Method | Purpose |
|--------|---------|
| `route(order, venues)` | Delegate to `SmartRouter` |
| `validate_order(order)` | `OrderValidator.validate` |
| `apply_event(event_id, ...)` | Idempotent ack / fill / cancel / reject |
| `reconcile(expected, executed, broker=None)` | Three-way position recon |

### Halt / kill

```python
engine.halt("market data gap", cancel_open=True)
engine.kill(scope="global", reason="manual")
engine.kill(scope="venue", key="NYSE", reason="venue outage")
engine.kill(scope="account", key="acct_1", reason="breach")
engine.kill(scope="strategy", key="strat_a", reason="model fault")
```

### Analytics, simulation, persistence

```python
engine.analytics(fills, arrival_price=190.0, side="buy", ordered_qty=1000)
engine.simulate_execution(side="buy", quantity=1000, mid=190.0, n_slices=5)
engine.save("/tmp/exec_state.json")
engine.load("/tmp/exec_state.json")
```

---

## `ExecutionReport`

| Field | Description |
|-------|-------------|
| `execution_id` | Unique run id |
| `status` | `FILLED` / `PARTIAL` / `COMPLETED` / `EMPTY` / `FAILED` / `BLOCKED` / … |
| `state` | Engine `ExecutionState` value |
| `algo` | Algorithm name used |
| `parents` / `children` / `fills` | Serialized entities |
| `routing` | `RoutingDecision.to_dict()` list |
| `pre_trade` / `post_trade` | Per-parent cost / IS attribution |
| `analytics` | `execution_quality_report` per parent |
| `latency` | `LatencyTracker.summary` |
| `audit` / `errors` / `metadata` | Trace + failures |

`report.to_dict()` is JSON-serializable.

---

## Hydra config

Path: `iqrp/configs/execution/default.yaml`

```yaml
seed: 42
data_version: "1.0.0"
model_version: "1.0.0"
default_time_in_force: DAY
default_urgency: NORMAL
default_venue: SIM

tick_lot:
  default_tick_size: 0.01
  default_lot_size: 1.0
  min_qty: 1.0
  max_qty: 1000000.0

price_bands:
  enabled: true
  band_pct: 0.10

capital:
  check_enabled: true
  max_notional: 10000000.0
  max_order_notional: 5000000.0

risk:
  enforce_hard_limits: true
  require_risk_callback: false
  max_participation: 0.10

kill_switch:
  check_on_submit: true
  check_global: true
  check_account: true
  check_venue: true
  check_strategy: true

reconciliation:
  qty_tolerance: 0.0
  notional_tolerance: 0.01
  alert_on_diff: true

fills:
  idempotent: true
  allow_overfill: false
```

Load:

```python
from iqrp.app.execution import ExecutionSettings

settings = ExecutionSettings.from_hydra()  # default.yaml
# or
settings = ExecutionSettings.default()
```

---

## Risk authority

1. **Pre-parent:** `_check_risk` on a probe order before slicing.  
2. **Validate:** `OrderValidator` (tick/lot/bands/capital + optional risk callback).  
3. **Submit:** kill-switch scopes + re-check `validate_risk` when `enforce_hard_limits`.  
4. **Route:** `SmartRouter` kill + optional `risk_check` per venue.

Urgency and algo aggression never skip these gates.

---

## Algorithm registry

```python
from iqrp.app.execution.registry import available_algorithms, get_algorithm

available_algorithms()
# ['adaptive', 'arrival', 'arrival_price', 'implementation_shortfall', 'is',
#  'limit', 'liquidity_seeking', 'market', 'opportunistic', 'pov', 'twap', 'vwap']

algo = get_algorithm("vwap", n_slices=8, participation_cap=0.15)
slices = algo.plan(10_000, {"mid": 100, "adv": 1e6, "volume_curve": [...]})
```

See [ExecutionAlgorithms](ExecutionAlgorithms.md), [TWAP](TWAP.md), [VWAP](VWAP.md), [POV](POV.md), [ImplementationShortfall](ImplementationShortfall.md).

---

## Latency and quality analytics

`LatencyTracker` records create → submit → ack → fill timestamps per order.  
`execution_quality_report` aggregates fill rate, arrival/VWAP/TWAP slippage, IS, and latency into the report’s `analytics` block.

Historical / synthetic paths without live venues:

```python
from iqrp.app.execution.simulation import simulate_execution

sim = simulate_execution(
    side="buy", quantity=5000, mid=100.0, spread=0.02, adv=1e6, n_slices=8
)
```

---

## Validation

```bash
python -m iqrp.app.execution.phase12
```

```python
from iqrp.app.execution.phase12 import validate_phase12, write_phase12_report

report = validate_phase12()
path = write_phase12_report()  # Phase12_ExecutionPlatform_Validation.json
```

Machine-readable report: [Phase12_ExecutionPlatform_Validation.json](Phase12_ExecutionPlatform_Validation.json).
