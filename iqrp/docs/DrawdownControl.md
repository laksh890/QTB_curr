# Drawdown Control

Drawdown analytics and risk-state mapping in `iqrp.app.risk.tail.drawdown`, driven by Hydra `drawdown` thresholds and enforced in `RiskIntelligenceEngine`.

Live **risk state** uses **current** underwater fraction so a recovered book can leave halt states. Historical max drawdown remains in the report for audit and hard limits.

---

## Tracked quantities

| Field | Description |
|-------|-------------|
| `current_drawdown` | `1 - wealth / peak` at latest bar |
| `peak_equity` | Running peak of wealth path |
| `drawdown_duration` | Bars since last peak (current episode) |
| `recovery_time` | Length of most recently **completed** DD episode (bars), or `None` |
| `max_drawdown` | Peak historical DD on the path |
| `wealth` | Current wealth (= cumprod(1+r)) |

Also reported under `measures`: max DD, expected (mean) DD, ulcer index, downside deviation.

```python
from iqrp.app.risk.tail.drawdown import (
    drawdown_series,
    drawdown_state,
    max_drawdown,
    expected_drawdown,
    ulcer_index,
    downside_deviation,
)

dd_path = drawdown_series(returns)
state = drawdown_state(
    returns,
    caution=0.05,
    reduced_risk=0.10,
    capital_preservation=0.15,
    trading_halt=0.20,
)
print(
    state["risk_state"],
    state["current_drawdown"],
    state["peak_equity"],
    state["drawdown_duration"],
    state["recovery_time"],
    state["max_drawdown"],
)
```

Engine:

```python
from iqrp.app.risk import RiskIntelligenceEngine, RiskSettings, RiskState

engine = RiskIntelligenceEngine(RiskSettings.default())
dd = engine.drawdown(returns)
assert RiskState(dd["risk_state"]) == engine.risk_state(returns)
```

---

## States: NORMAL → TRADING_HALT

Deterministic mapping on **current** drawdown vs thresholds:

```text
current_dd < caution              → NORMAL
caution ≤ dd < reduced_risk       → CAUTION
reduced_risk ≤ dd < capital_pres  → REDUCED_RISK
capital_pres ≤ dd < trading_halt  → CAPITAL_PRESERVATION
dd ≥ trading_halt                 → TRADING_HALT
```

```python
from iqrp.app.risk import RiskState

# Enum
RiskState.NORMAL
RiskState.CAUTION
RiskState.REDUCED_RISK
RiskState.CAPITAL_PRESERVATION
RiskState.TRADING_HALT
```

Hydra (`configs/risk/default.yaml`):

```yaml
drawdown:
  caution: 0.05
  reduced_risk: 0.10
  capital_preservation: 0.15
  trading_halt: 0.20
```

---

## Configurable responses

| State | Typical response (composition policy) |
|-------|----------------------------------------|
| `NORMAL` | Full sizing within hard limits |
| `CAUTION` | Tighten confidence floors; prefer vol-target reductions |
| `REDUCED_RISK` | `regime`/`drawdown_adjusted` cuts; soft escalations |
| `CAPITAL_PRESERVATION` | Aggressive size/leverage cuts; HARD alerts |
| `TRADING_HALT` | **`validate_position` rejects** all new risk; leverage → min |

Built-in enforcement examples:

```python
# Halt blocks new positions
decision = engine.validate_position(
    proposed_weight=0.05,
    weights=weights,
    returns=returns_in_halt,
)
# approved=False, reason contains TRADING_HALT

# Sizing collapses as DD → trading_halt
engine.position_size(
    realized_vol=0.1,
    current_drawdown=0.18,
    method="drawdown_adjusted",
)

# Leverage hard-halts at max_drawdown
engine.recommended_leverage(
    realized_vol=0.1,
    current_drawdown=0.20,  # ≥ settings.drawdown.trading_halt
)
# → min_leverage regardless of confidence
```

Wire desk-specific playbooks (hedge, flatten, notify) at the portfolio/execution composition root using `risk_state()` — Risk supplies the state machine and audit trail.

---

## Monitoring

```python
snap = engine.monitor_snapshot()
# alerts escalate with state: CAUTION→WARNING, REDUCED_RISK→SOFT,
# CAPITAL_PRESERVATION/TRADING_HALT→HARD
```

Transitions are logged via `RiskDecision.audit` and `export_state()["audit_log"]` when validation runs.

---

## Invariants

1. State transitions are threshold-based, deterministic, and configurable.  
2. Hard DD limit = `drawdown.trading_halt`; confidence cannot relax it.  
3. Point-in-time wealth path only — no future returns in DD calculation.
