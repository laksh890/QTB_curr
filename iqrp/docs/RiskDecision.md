# Risk Decision

Pre-trade / exposure decisions produced by the Risk Intelligence Ensemble gate.

Package: `iqrp.app.risk.ensemble`  
Modules: `decision.py`, `risk_ensemble.py`  
Types: `DecisionAction`, `EnsembleDecision`

Related: [Risk Ensemble](RiskEnsemble.md) · [Risk State Machine](RiskStateMachine.md) · [Risk Scoring](RiskScoring.md) · [Risk Framework](RiskFramework.md)

---

## `DecisionAction`

```python
from iqrp.app.risk.ensemble import DecisionAction

DecisionAction.APPROVE
DecisionAction.APPROVE_REDUCED
DecisionAction.REJECT
DecisionAction.HALT
```

| Action | Meaning |
|--------|---------|
| `APPROVE` | Proposed exposure within caps for current state |
| `APPROVE_REDUCED` | Allowed only at reduced size / under elevated state |
| `REJECT` | Do not take the proposed risk (hard reject or over-cap) |
| `HALT` | Trading halt — no new risk (`TRADING_HALT` or engine halt) |

---

## `EnsembleDecision` fields

```python
from iqrp.app.risk.ensemble import EnsembleDecision

# decision: DecisionAction
# risk_state: RiskState
# risk_score: RiskScore
# risk_confidence: float          # ensemble confidence, not forecast confidence
# triggered_limits: list[str]
# reasons: list[str]              # every rejection / reduction is explicit
# required_position_reduction: float   # [0, 1]
# maximum_permitted_exposure: float
# recommended_leverage: float
# timestamp / data_version / model_versions
# audit: dict                     # includes hard_reject, state_cap, engine payload
# proposed_exposure: float
# forecast_confidence: float      # recorded; cannot override hard limits
```

Serialize with `decision.to_dict()` for audit logs.

---

## How actions are chosen

`build_decision` / `action_for_state`:

1. Resolve state caps (`max_exposure`, `recommended_leverage`, `position_reduction`).
2. Cap exposure further by `limits.max_gross_exposure`.
3. If assessment fallback applied → use `missing_metrics_fallback_action` (default `REJECT`).
4. If `hard_reject` from engine → `REJECT`, or `HALT` when state is `TRADING_HALT`.
5. `TRADING_HALT` → `HALT`.
6. `CAPITAL_PRESERVATION` → `APPROVE_REDUCED` (or `REJECT` if proposed exposure exceeds the state max).
7. Proposed exposure above max → `APPROVE_REDUCED` in caution/reduced states, else `REJECT`.
8. `REDUCED_RISK` with positive exposure → prefer `APPROVE_REDUCED`.
9. Else → `APPROVE`.

Absolute guard: missing critical metrics can never yield a final `APPROVE`.

---

## Forecast confidence cannot override

Hard institutional rule (framework + ensemble):

> Forecast confidence / Kelly **must not** override hard risk limits or halt states.

Implementation:

```python
# apply_confidence_within_caps — leverage only
# conf scales inside [1/confidence_cap, confidence_cap]
# then clip to state recommended_leverage and settings.leverage.max_leverage
# TRADING_HALT → min_leverage (typically 0)
```

When `forecast_confidence > 0` under `TRADING_HALT` or `CAPITAL_PRESERVATION`, reasons explicitly record that confidence cannot override the hard state. The decision field stores the forecast confidence for audit only.

```python
decision = ens.decision(
    metrics=metrics,
    proposed_exposure=1.2,
    forecast_confidence=0.99,  # may raise leverage slightly inside caps only
)
# Never expands maximum_permitted_exposure beyond state_caps
# Never turns HALT/REJECT into APPROVE
```

---

## Position validation flow

`RiskIntelligenceEnsemble.validate_position` is the pre-trade entry point when the ensemble is the gate:

```text
proposed_weight + current weights
        │
        ▼
proposed_exposure = Σ |w|
        │
        ▼
enrich metrics (vol, VaR, CVaR, drawdown) — PIT, no look-ahead
        │
        ▼
missing critical? → logged; aggregate still runs with fallback
        │
        ▼
optional RiskIntelligenceEngine.validate_position
        │
        ├─ approved=False → hard_reject (engine wins)
        └─ engine TRADING_HALT → force ensemble halt
        │
        ▼
build EnsembleDecision (state caps + confidence-within-caps)
        │
        ▼
re-apply exposure / leverage caps → return EnsembleDecision
```

```python
from iqrp.app.risk.ensemble import RiskIntelligenceEnsemble
from iqrp.app.risk import RiskIntelligenceEngine, RiskSettings

ens = RiskIntelligenceEnsemble(
    risk_engine=RiskIntelligenceEngine(RiskSettings.default())
)

d = ens.validate_position(
    proposed_weight=0.08,
    weights=weights,
    returns=returns,
    metrics={"liquidity_score": 0.8, "concentration": 0.12},
    forecast_confidence=0.85,
    realized_vol=0.20,
    participation=0.05,
    adv_coverage=10.0,
    asset_index=0,
)

assert d.decision.value in {"APPROVE", "APPROVE_REDUCED", "REJECT", "HALT"}
if d.decision.value != "APPROVE":
    # use d.reasons, d.required_position_reduction, d.maximum_permitted_exposure
    ...
```

---

## Decision construction without full validate

```python
assessment = ens.aggregate(metrics)
decision = ens.decision(
    assessment=assessment,
    proposed_exposure=0.6,
    forecast_confidence=0.5,
    triggered_limits=["max_concentration"],
    hard_reject=False,
)
```

Or pass `metrics=` and let `decision` call `aggregate` internally.

---

## Audit expectations

Every decision must be reproducible from:

- recorded metrics / assessment timestamp
- `data_version` / `model_versions` / `ensemble_version`
- state caps and limits settings
- `hard_reject` + engine audit blob
- explicit `reasons` and `triggered_limits`

Framework rule: **no trading component may bypass** `validate_position` / `check_limits` on the live path.
