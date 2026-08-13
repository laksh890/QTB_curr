# Risk State Machine

Deterministic `RiskState` transitions with hysteresis, recovery confirmations, and multi-dimension halt confirmation.

Package: `iqrp.app.risk.ensemble`  
Module: `state_machine.py`  
Type: `EnsembleStateMachine`  
Shared enum: `iqrp.app.risk.base.RiskState`

Related: [Risk Ensemble](RiskEnsemble.md) · [Risk Scoring](RiskScoring.md) · [Risk Decision](RiskDecision.md) · [Drawdown Control](DrawdownControl.md)

---

## State ladder

```text
NORMAL → CAUTION → REDUCED_RISK → CAPITAL_PRESERVATION → TRADING_HALT
```

Recovery walks the ladder **downward**, one step per confirmed recovery cycle, subject to hysteresis thresholds.

| State | Typical meaning | Default exposure / leverage caps* |
|-------|-----------------|-----------------------------------|
| `NORMAL` | Risk within tolerance | max exposure 1.0, lev 1.0 |
| `CAUTION` | Elevated; tighten | 0.75 / 0.75, reduce 25% |
| `REDUCED_RISK` | Material stress | 0.50 / 0.50, reduce 50% |
| `CAPITAL_PRESERVATION` | Severe; preserve capital | 0.25 / 0.25, reduce 75% |
| `TRADING_HALT` | No new risk | 0.0 / 0.0, reduce 100% |

\*From `EnsembleSettings.state_caps` / Hydra `state_caps`.

---

## Escalation from scores

Candidate state is derived from `RiskScore.overall` vs `state_thresholds`:

| Threshold key | Default overall ≥ |
|---------------|-------------------|
| `caution` | 0.35 |
| `reduced_risk` | 0.55 |
| `capital_preservation` | 0.72 |
| `trading_halt` | 0.88 |

```python
from iqrp.app.risk.ensemble import RiskIntelligenceEnsemble
from iqrp.app.risk.base import RiskState

ens = RiskIntelligenceEnsemble()
state = ens.risk_state(scores, previous_state=RiskState.NORMAL)
```

Escalation requires `hysteresis.escalation_confirmations` consecutive evaluations targeting the same higher state (default **1**).

---

## Multi-dimension halt confirmation

`TRADING_HALT` is special. A single noisy hot metric must **not** halt trading by default.

| Setting | Default | Role |
|---------|---------|------|
| `hard_halt_on_single` | **`false`** | If `true`, overall halt threshold alone may halt |
| `min_dimensions_for_halt` | `2` | Minimum “hot” dimensions required when `hard_halt_on_single` is false |
| `hysteresis.dimension_confirmation_threshold` | `0.75` | Dimension score ≥ this counts as hot |

When overall score candidates for halt but confirmation fails, the raw target is **downgraded to `CAPITAL_PRESERVATION`** and the history records `halt_confirmation: blocked_insufficient_dimensions`.

```yaml
# iqrp/configs/risk/ensemble/default.yaml
hard_halt_on_single: false
min_dimensions_for_halt: 2
hysteresis:
  escalation_confirmations: 1
  recovery_confirmations: 3
  dimension_confirmation_threshold: 0.75
```

---

## Recovery and hysteresis

Recovery uses **lower** `recovery_thresholds` so the book does not chatter around the escalation boundary:

| Threshold key | Default overall ≥ (recovery ceiling) |
|---------------|--------------------------------------|
| `caution` | 0.28 |
| `reduced_risk` | 0.45 |
| `capital_preservation` | 0.62 |
| `trading_halt` | 0.78 |

Rules:

1. Overall must clear the recovery ceiling for a lower state.
2. Descent is **at most one ladder step** per confirmed cycle.
3. Needs `hysteresis.recovery_confirmations` consecutive confirmations (default **3**).
4. Pending escalation and recovery counters reset when direction flips.
5. History entries record `escalated`, `escalation_pending`, `recovered`, `recovery_pending`, `recovery_blocked_hysteresis`, or `hold`.

---

## API

```python
from iqrp.app.risk.ensemble.state_machine import EnsembleStateMachine
from iqrp.app.risk.ensemble import EnsembleSettings
from iqrp.app.risk.base import RiskState

sm = EnsembleStateMachine(EnsembleSettings.default())
state = sm.transition(risk_score)                 # updates current_state
state = sm.transition(risk_score, previous_state=RiskState.CAUTION)
state = sm.transition(risk_score, force_state=RiskState.TRADING_HALT)

sm.reset(RiskState.NORMAL)
payload = sm.export_state()
sm.import_state(payload)
sm.history  # audit trail of transitions
```

`force_state` clears streaks and is used when the underlying `RiskIntelligenceEngine` forces a halt.

---

## Interaction with drawdown and capital

Drawdown thresholds in capital / ensemble configs (`caution` 5% → `trading_halt` 20%) feed **scoring** and capital scaling. The state machine itself consumes the ensemble `RiskScore.overall` and hot dimensions — not raw PnL — so multi-dimension confirmation still applies before halt.

Capital allocation maps risk states to scales (`NORMAL: 1.0` … `TRADING_HALT: 0.0`) independently; see [Capital Allocation](CapitalAllocation.md).

---

## Guarantees

- Transitions are **deterministic** given scores, settings, and machine counter state.
- Default **`hard_halt_on_single=false`** — single-dimension noise cannot alone produce `TRADING_HALT`.
- Forecast confidence has **no direct state transition path**.
- Persisted counters (`export_state` / `import_state`) keep hysteresis coherent across process restarts when the host wires them.
