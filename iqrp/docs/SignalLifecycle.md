# Signal Lifecycle

Auditable lifecycle for alpha research experiments: from `CANDIDATE` through research and validation to `APPROVED`, with degradation, retirement, and rejection paths that preserve history.

**Package:** `iqrp.app.alpha.base.signal_result` · `iqrp.app.alpha.base.signal_registry` · `iqrp.app.alpha.monitoring`  
**Engine entry:** `register` / `approve` / `degrade` / `retire`  
**Parent:** [AlphaResearch](AlphaResearch.md) · Related: [SignalRetirement](SignalRetirement.md) · [SignalMonitoring](SignalMonitoring.md) · [ExperimentRegistry](ExperimentRegistry.md)

---

## Status model

```text
CANDIDATE → RESEARCHING → VALIDATING → PROVISIONAL → APPROVED
    │            │              │            │            │
    │            │              │            │            ├→ DEGRADED ⇄ PROVISIONAL / VALIDATING
    │            │              │            │            └→ RETIRED
    │            │              │            ├→ DEGRADED / REJECTED / RETIRED
    │            │              └→ PROVISIONAL / REJECTED / RESEARCHING / RETIRED
    │            └→ VALIDATING / REJECTED / CANDIDATE / RETIRED
    └→ RESEARCHING / REJECTED / RETIRED

REJECTED and RETIRED are terminal (no further promotion).
```

| Status | Meaning |
|--------|---------|
| `CANDIDATE` | Discovered / registered lead; no profitability claim |
| `RESEARCHING` | Evaluation underway |
| `VALIDATING` | Statistical / CV evidence being attached |
| `PROVISIONAL` | Passed research gates pending final approval |
| `APPROVED` | Research-approved alpha (**≠ trading approval**) |
| `DEGRADED` | Live/research performance impaired; may recover or retire |
| `RETIRED` | Terminal — removed from active research/trading consideration |
| `REJECTED` | Terminal — failed research; **preserved** in registry |

Illegal transitions raise `ValueError` via `validate_transition`.

```python
from iqrp.app.alpha import SignalStatus, validate_transition

validate_transition(SignalStatus.CANDIDATE, SignalStatus.RESEARCHING)  # ok
# validate_transition(SignalStatus.REJECTED, SignalStatus.APPROVED)  # raises
```

---

## Auditable transitions

Every change records `StatusTransition`: `from_status`, `to_status`, `reason`, `timestamp`, `actor`, optional `extras`.

```python
from iqrp.app.alpha import AlphaResearchEngine, SignalDefinition, SignalStatus

eng = AlphaResearchEngine()
defn = SignalDefinition(
    name="demo",
    version="0.1.0",
    formula="x",
    features=("x",),
    lookback=5,
    horizon=1,
    universe="default",
    frequency="1d",
    direction="long_short",
    expected_relationship="positive",
    economic_hypothesis=(
        "Demo hypothesis with sufficient length for APPROVED gates later."
    ),
    owner="research",
)
rec = eng.register(defn, signal=None, reason="initial registration", actor="analyst")

eng.registry.transition(
    rec.experiment_id,
    SignalStatus.RESEARCHING,
    reason="started IC evaluation",
    actor="analyst",
)
trail = eng.registry.audit_trail(rec.experiment_id)
assert trail[-1].reason == "started IC evaluation"
```

Hydra: `governance.auditable_transitions: true`. Empty `reason` is rejected.

---

## Promotion path (`approve`)

`AlphaResearchEngine.approve` walks allowed steps:

`CANDIDATE → RESEARCHING → VALIDATING → PROVISIONAL → APPROVED`

Hard gates:

1. Substantive `economic_hypothesis` (≥ `min_hypothesis_chars`)
2. Not Sharpe-only (`allow_sharpe_only_approval=false` by default)
3. Evaluate + validate evidence present on the report

```python
import numpy as np

sig = np.random.default_rng(0).normal(size=400)
ret = np.random.default_rng(1).normal(0, 0.01, 400)
rec = eng.register(defn, signal=sig)
eng.evaluate(sig, ret, experiment_id=rec.experiment_id, definition=defn)
eng.validate(sig, ret, experiment_id=rec.experiment_id, n_trials=20)

approved = eng.approve(
    rec.experiment_id,
    reason="BH-FDR IC ok; DSR/PBO acceptable; capacity within ADV budget",
    actor="lead_researcher",
)
assert approved.status == SignalStatus.APPROVED
assert approved.report.diagnostics.get("approval_extras", {}) or True
# extras include risk_intelligence_not_bypassed / alpha_approval_is_not_trading_approval
```

Registry also refuses `APPROVED` without hypothesis even on direct `transition`.

---

## Degrade and retire

```python
# Post-approval impairment
eng.degrade(approved.experiment_id, reason="rolling IC collapsed vs baseline", actor="monitor")

# Terminal retirement
eng.retire(approved.experiment_id, reason="cost dominance at target AUM", actor="pm")
```

Engine nuances:

| Call | Behavior |
|------|----------|
| `degrade` on pre-approval statuses | Routes to `REJECTED` (research path failure) |
| `degrade` on `PROVISIONAL` / `APPROVED` | → `DEGRADED` |
| `retire` from most non-terminal | Direct → `RETIRED` when allowed |
| `retire` on `REJECTED` | Raises `ApprovalError` |

---

## Retirement triggers (monitoring)

`evaluate_retirement` recommends `ACTIVE` / `DEGRADED` / `RETIRED` from research metrics — callers then perform the registry transition.

```python
from iqrp.app.alpha.monitoring.retirement import evaluate_retirement, batch_evaluate_retirement

rec_decision = evaluate_retirement(
    ic_recent=0.005,
    ic_baseline=0.04,
    net_sharpe=-0.2,
    gross_sharpe=0.8,
    cost_ratio=0.85,
    capacity=2e6,
    capacity_baseline=2e7,
    drift_severity="high",
    regime_unstable=False,
    performance_decayed=True,
)
# reasons may include: ic_collapse, cost_dominance, capacity_collapse,
# model_drift, regime_instability, performance_decay, net_sharpe_negative, ...

if rec_decision["recommend"] == "RETIRED":
    eng.retire(eid, reason=",".join(rec_decision["reasons"]), actor="monitor")
elif rec_decision["recommend"] == "DEGRADED":
    eng.degrade(eid, reason=",".join(rec_decision["reasons"]), actor="monitor")
```

Default thresholds (overridable):

| Key | Default | Effect |
|-----|---------|--------|
| `ic_collapse_ratio` | 0.25 | Recent \|IC\| / baseline below → collapse |
| `ic_degrade_ratio` | 0.50 | Softer IC degradation |
| `net_sharpe_retire` | 0.0 | Below → retire pressure |
| `net_sharpe_degrade` | 0.3 | Weak net Sharpe |
| `cost_dominance` | 0.80 | Costs consume edge |
| `capacity_collapse_ratio` | 0.30 | Capacity vs baseline |
| `capacity_degrade_ratio` | 0.60 | Softer capacity hit |

Complementary monitors: `monitor_ic_decay`, `monitor_performance_decay`, `monitor_signal_drift`, `build_alpha_alerts`.

---

## Monitoring loop

```python
from iqrp.app.alpha.monitoring.alerts import build_alpha_alerts
from iqrp.app.alpha.monitoring.signal_decay import rolling_ic, monitor_ic_decay

roll = rolling_ic(signal, forward_returns, window=60)
decay = monitor_ic_decay(roll, baseline_ic=0.04)
alerts = build_alpha_alerts(
    {
        "demo": {
            "ic_status": decay["status"],
            "net_sharpe": -0.1,
            # additional alert fields per helper
        }
    }
)
```

Monitoring never auto-bypasses Risk Intelligence. Alerts recommend lifecycle actions; humans or orchestration call `degrade` / `retire`.

---

## Rejected experiments are preserved

```python
from iqrp.app.alpha import get_default_registry, SignalStatus

reg = get_default_registry()
reg.transition(eid, SignalStatus.REJECTED, reason="failed BH-FDR after nested CV", actor="research")

assert reg.get(eid).rejected is True
assert eid in {r.experiment_id for r in reg.rejected_experiments()}
# list_experiments(include_rejected=True) by default — governance.preserve_rejected
```

Do not delete rejects to “clean” the registry. Trial accounting, institutional learning, and false-discovery audit depend on retention.

---

## End-to-end sketch

```python
from iqrp.app.alpha import AlphaResearchEngine, AlphaSettings

eng = AlphaResearchEngine(AlphaSettings.default())
cands = eng.discover(returns=returns, features=features)
# pick / refine definition → register → evaluate → validate
# → analyze_decay / analyze_capacity → compare/rank
# → approve (research) → monitor → degrade/retire as needed
# → trading still requires Risk Intelligence + portfolio gates
```
