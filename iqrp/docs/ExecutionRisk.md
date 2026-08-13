# Execution Risk

Hard pre-trade checks, Risk Intelligence authority, halt semantics, and kill switches for the Institutional Execution Platform.

**Types:** `KillSwitch` (`iqrp.app.execution.types`) · `OrderValidator` · `ExecutionEngine.halt` / `.kill`  
**Config:** `risk` / `kill_switch` / `capital` / `price_bands` in `configs/execution/default.yaml`

Related: [ExecutionPlatform](ExecutionPlatform.md) · [OrderManager](OrderManager.md) · [SmartRouting](SmartRouting.md) · [OrderLifecycle](OrderLifecycle.md)

---

## Principles

1. **Risk Intelligence is authoritative** when a `risk_engine` or `validate_risk` callback is provided.  
2. **Kill switches are fail-safe** — engaged scopes block submit and routing.  
3. **Urgency never overrides** hard risk, capital, bands, or kill switches.  
4. **Fail closed** on risk-engine errors when `enforce_hard_limits=True`.  
5. **On HALT:** stop new orders; optionally cancel open working orders.

---

## Pre-trade checks

### OrderValidator

Validates microstructure and capital before approval:

- Instrument trading enabled / known meta  
- Quantity within min/max and lot size  
- Limit price on tick; optional price band vs reference (`band_pct`)  
- Notional vs `capital.max_order_notional` / `max_notional`  
- Optional `validate_risk(order) -> (ok, reason)` hard gate  

```python
from iqrp.app.execution import OrderManager, Side, OrderType

def risk_cb(order):
    if order.quantity > 50_000:
        return False, "size limit"
    return True, ""

om = OrderManager(validate_risk=risk_cb)
# om.validate_and_approve(...)  → REJECTED + ValidationError if risk_cb fails
```

### ExecutionEngine + Risk Intelligence

```python
class StubRisk:
    def validate_position(self, positions: dict):
        if abs(sum(positions.values())) > 1e6:
            return False, "gross limit"
        return True, ""

engine = ExecutionEngine(risk_engine=StubRisk())
```

Adapter order: prefer `validate_position`, then `check_limits`. Supports tuple `(ok, reason)`, objects with `.approved`, or dicts with `ok`/`approved`. Exceptions under `enforce_hard_limits` → hard reject.

Checks run:

1. Before algorithm slicing (probe order)  
2. Inside `validate_and_approve`  
3. Immediately before `submit` (re-check)

### Participation

`settings.risk.max_participation` and algo-level caps (POV/VWAP/TWAP) bound ADV participation. Urgency may raise *target* rates but not past hard maxes.

---

## Kill switches

```python
from iqrp.app.execution import KillSwitch

ks = KillSwitch()
ks.engage_global("ops incident")
ks.engage_account("acct_1", "margin")
ks.engage_venue("NYSE", "halt")
ks.engage_strategy("strat_a", "model fault")

blocked, reason = ks.is_blocked(account_id="acct_1", venue="NYSE", strategy_id="strat_a")
# Aliases: halt_global / halt_account / halt_venue / halt_strategy
# Clear: clear_global / clear_account / clear_venue / clear_strategy
print(ks.to_dict())  # includes audit trail
```

### Config (`kill_switch` section)

| Flag | Default | Effect |
|------|---------|--------|
| `check_on_submit` | true | Gate `OrderManager.submit` |
| `check_global` | true | Honor `global_halt` |
| `check_account` | true | Scope by `account_id` |
| `check_venue` | true | Scope by venue |
| `check_strategy` | true | Scope by `strategy_id` |

`SmartRouter` also consults the shared `KillSwitch` before accepting a venue.

---

## Engine halt / kill

```python
from iqrp.app.execution import ExecutionEngine

engine = ExecutionEngine()

engine.halt("feed stale", cancel_open=True)
# - _halted = True, state = HALTED
# - engage_global on kill_switch
# - cancel non-terminal open orders when cancel_open=True
# - subsequent execute/plan/route raise ExecutionError(EXECUTION_HALTED | KILL_SWITCH_ACTIVE)

engine.kill(scope="global", reason="manual")
engine.kill(scope="venue", key="SIM", reason="venue down")
engine.kill(scope="account", key="acct_1", reason="breach")
engine.kill(scope="strategy", key="s1", reason="kill")
```

Global kill also sets engine halted. Scoped kills block matching orders without necessarily cancelling all books (use `halt(..., cancel_open=True)` for broad cancel).

---

## Failure handling

| Condition | Result |
|-----------|--------|
| Kill / halt during `execute` | Raise `ExecutionError`; status path `BLOCKED` / halted |
| Hard risk reject | `HARD_RISK_REJECT` — no children submitted |
| Validation failure | `ORDER_VALIDATION_FAILED` — order `REJECTED` |
| Routing reject | Child skipped; error recorded; other slices may continue |
| Venue reject | Audited; no fill invented |
| Illegal state transition | `ORDER_STATE_TRANSITION_ILLEGAL` / execution equivalent |

```python
from iqrp.app.core.exceptions import ExecutionError

ks = KillSwitch()
ks.engage_global("test")
engine = ExecutionEngine(kill_switch=ks)
try:
    engine.execute({"AAPL": 100}, current={"AAPL": 0}, algo="twap",
                   market_context={"AAPL": {"mid": 190, "adv": 1e6}})
except ExecutionError as e:
    assert e.code == "KILL_SWITCH_ACTIVE"
```

---

## What urgency may and may not do

| Allowed | Forbidden |
|---------|-----------|
| Fewer/larger slices | Bypass kill switch |
| Higher POV *target* (≤ max) | Exceed approved residual |
| More aggressive limit hints | Skip `validate_risk` |
| Front-load IS schedule | Relax capital / price bands |

---

## Audit

- `KillSwitch.audit` — engage/clear events with timestamps  
- `OrderManager.audit` / `order.audit` — validation rejects, kill fails, fills  
- `ExecutionEngine` audit / `ExecutionReport.audit` — halt, kill, pre-trade, submit  

Persisted via `engine.save` / `load` including kill-switch snapshot and processed event ids.
