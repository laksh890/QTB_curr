# Position Reconciliation

Three-way comparison of **expected** (approved targets / books), **executed** (fills applied in Order Manager), and **broker** (custodian / exchange positions), with severity-graded alerts on material diffs.

**Types:** `PositionReconciler`, `ReconciliationResult`, `ReconciliationAlert`, `PositionSnapshot`  
**Package:** `iqrp.app.execution.order_manager.position_reconciliation`

Related: [OrderManager](OrderManager.md) · [ExecutionPlatform](ExecutionPlatform.md) · [ExecutionRisk](ExecutionRisk.md)

---

## Rules

1. Never invent fills or future positions to force a match.  
2. Use only observed quantities at reconciliation time (point-in-time).  
3. Alert on diffs when `alert_on_diff=True`; do not silently coerce books.  
4. Execution never generates alpha — recon is diagnostic / control, not a trading signal.

---

## Configuration

From `configs/execution/default.yaml`:

```yaml
reconciliation:
  qty_tolerance: 0.0
  notional_tolerance: 0.01
  alert_on_diff: true
```

| Setting | Role |
|---------|------|
| `qty_tolerance` | Absolute qty band for “matched” |
| `notional_tolerance` | Reserved for notional-aware checks |
| `alert_on_diff` | Emit `ReconciliationAlert` when outside tolerance |

---

## API

```python
from iqrp.app.execution import OrderManager, ExecutionEngine

om = OrderManager()
result = om.reconcile_positions(
    expected={"AAPL": 1000.0, "MSFT": 200.0},
    executed={"AAPL": 1000.0, "MSFT": 200.0},
    broker={"AAPL": 1000.0, "MSFT": 198.0},
)
print(result.matched, len(result.alerts))
print(result.per_instrument["MSFT"])
# {'expected': 200.0, 'executed': 200.0, 'broker': 198.0,
#  'expected_vs_executed': 0.0, 'executed_vs_broker': -2.0, 'expected_vs_broker': -2.0}

# Engine facade (broker defaults to executed if omitted)
engine = ExecutionEngine()
result = engine.reconcile(
    expected={"AAPL": 1000},
    executed={"AAPL": 995},
    broker={"AAPL": 995},
)
```

Direct constructor:

```python
from iqrp.app.execution import PositionReconciler

recon = PositionReconciler(qty_tolerance=1.0, alert_on_diff=True)
result = recon.reconcile(
    expected={"XYZ": 500},
    executed={"XYZ": 500},
    broker={"XYZ": 500},
)
assert result.matched
```

---

## Diff math

For each instrument in the union of keys (uppercased in output):

| Diff | Formula |
|------|---------|
| `expected_vs_executed` | executed − expected |
| `executed_vs_broker` | broker − executed |
| `expected_vs_broker` | broker − expected |

If `max(|diffs|) ≤ qty_tolerance` → no alert for that name.  
Otherwise severity:

| Severity | Condition |
|----------|-----------|
| `INFO` | Outside tolerance but ≤ tolerance + 1.0 |
| `WARNING` | Larger material break |
| `CRITICAL` | `max(|diff|) > max(tolerance, 1) × 10` |

`ReconciliationResult.matched` is `True` only when **no** alerts were produced.

---

## Alert payload

```python
for alert in result.alerts:
    print(alert.severity, alert.message)
    # alert.to_dict() → instrument, severity, quantities, three deltas, timestamp
```

`PositionReconciler.alerts` retains a running history of emitted alerts for the instance.

---

## Typical control flow

```text
Portfolio approved targets  ──►  expected
OrderManager fills          ──►  executed
Broker / custodian snapshot ──►  broker
                │
                ▼
        PositionReconciler
                │
        matched? ──no──► ops alert / halt investigation
           │
          yes
           ▼
        continue / archive
```

On persistent `executed_vs_broker` breaks, operators may `engine.halt` or engage a strategy/account kill switch ([ExecutionRisk](ExecutionRisk.md)) — reconciliation itself does not auto-trade.

---

## Snapshots (optional typing)

`PositionSnapshot(instrument, quantity, source, timestamp, notional=None)` documents a single book observation (`source` ∈ `expected` | `executed` | `broker`). The reconciler API accepts plain `dict[str, float]` maps for operational simplicity.
