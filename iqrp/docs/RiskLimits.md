# Risk Limits

Hierarchical limit framework under `iqrp.app.risk.limits` and `iqrp.app.risk.base.risk_limits`.

Hard limits **cannot** be overridden by forecast confidence. `check_all_limits` accepts no confidence override parameter.

---

## Hierarchical scopes

Limits carry a `scope` string used for aggregation, alerts, and audit:

| Scope | Typical limits |
|-------|----------------|
| `position` / asset | `max_position`, `max_single_name`, participation, ADV coverage, TTL |
| `strategy` | Strategy-level caps via custom `RiskLimit(scope="strategy")` |
| `portfolio` | Gross/net exposure, concentration, daily loss, drawdown |
| `desk` | Desk books — attach via metadata / hierarchical aggregate |
| `account` | Account NAV / margin-linked thresholds |
| `system` | Global leverage / trading-halt drawdown |

```python
from iqrp.app.risk.base import RiskLimit, LimitSeverity

desk_cap = RiskLimit(
    name="max_desk_gross",
    threshold=3.0,
    severity=LimitSeverity.HARD,
    scope="desk",
)
```

Aggregate nested books with `iqrp.app.risk.aggregation.hierarchical_aggregate`.

---

## Severities: hard / soft / warning

```python
from iqrp.app.risk import LimitSeverity

# WARNING — advisory
# SOFT    — approve with warnings / escalate
# HARD    — reject; never overridden by confidence
```

Default builders:

| Limit | Default severity |
|-------|------------------|
| `max_position`, exposures, `max_daily_loss`, `max_drawdown`, `max_concentration`, `max_participation`, `min_adv_coverage` | **HARD** |
| `max_herfindahl`, `max_time_to_liquidate` | **SOFT** (when built via defaults) |

`max_drawdown` is **always HARD** even if a softer severity is requested for other loss limits.

---

## Escalation

```text
WARNING  → log / dashboard
SOFT     → APPROVED_WITH_WARNINGS (validate_position)
HARD     → REJECTED + audit reason
RiskState CAUTION → REDUCED_RISK → CAPITAL_PRESERVATION → TRADING_HALT
         (monitor alerts escalate severity with state)
```

`build_alerts` prioritizes `HARD > SOFT > WARNING` and elevates alerts when `RiskState` rises.

```python
from iqrp.app.risk.monitoring.alerts import build_alerts

alerts = build_alerts(breaches=breaches, risk_state=state)
```

---

## Building and checking limits

```python
from iqrp.app.risk.limits import build_default_limits, check_all_limits
from iqrp.app.risk import RiskIntelligenceEngine, RiskSettings

limits = build_default_limits(
    max_position=0.10,
    max_gross_exposure=1.5,
    max_net_exposure=1.0,
    max_concentration=0.25,
    max_daily_loss=0.03,
    max_drawdown=0.20,
    max_participation=0.10,
    min_adv_coverage=0.01,
)

breaches = check_all_limits(
    weights=weights,
    daily_loss=0.01,
    current_drawdown=0.04,
    participation=0.05,
    adv_coverage=20.0,
)

engine = RiskIntelligenceEngine(RiskSettings.default())
breaches = engine.check_limits(
    weights=weights,
    daily_loss=0.01,
    current_drawdown=0.04,
    participation=0.05,
    adv_coverage=20.0,
)
```

Hydra (`configs/risk/default.yaml` → `limits` + `drawdown.trading_halt` for max DD).

---

## validate_position gate — steps 1–9

Every proposed position must pass the following. Implemented primarily in `RiskIntelligenceEngine.validate_position` + `check_limits` / drawdown / leverage / sizing.

| Step | Check | Mechanism |
|------|--------|-----------|
| **1. Position validation** | `|w_i| ≤ max_position` | `check_positions` / position limits |
| **2. Exposure validation** | Gross / net ≤ caps | `check_exposure_limits` |
| **3. Liquidity validation** | Participation ≤ cap; ADV coverage ≥ min | `check_liquidity_limits` when metrics supplied |
| **4. Concentration validation** | Max weight / HHI | `check_concentration_limits` |
| **5. Portfolio risk validation** | Book consistency after proposed weight splice | Weights updated then re-checked; portfolio measures available via `calculate_risk` |
| **6. Drawdown validation** | Current DD vs halt; `RiskState` | `drawdown()` + `TRADING_HALT` reject |
| **7. Leverage validation** | `sum(|w|) ≤ max_leverage` | Explicit hard breach appended in `validate_position` |
| **8. Model confidence validation** | Confidence scales **recommended size only** | Passed into `position_size` / `recommended_leverage`; **cannot clear HARD breaches** |
| **9. Global risk limits** | Combined hard breaches + halt state | Any HARD → `approved=False` with explicit reason |

```python
decision = engine.validate_position(
    proposed_weight=0.12,
    weights=current_weights,
    returns=pit_returns,
    realized_vol=0.15,
    participation=0.08,
    adv_coverage=15.0,
    forecast_confidence=0.99,  # informational; cannot override HARD
    asset_index=0,
)

if not decision.approved:
    # e.g. "REJECTED: hard limit breach(es); forecast confidence=0.99 cannot override hard risk limits. ..."
    raise RuntimeError(decision.reason)

# Soft-only → approved=True, reason starts with APPROVED_WITH_WARNINGS
```

Decision payload includes `breaches`, `risk_state`, `recommended_size`, `recommended_leverage`, and full `audit` dict.

---

## Custom evaluation

```python
from iqrp.app.risk.base import evaluate_limits

breaches = evaluate_limits(limits, {"max_position": 0.15, "max_daily_loss": 0.01})
```

---

## Design invariants

1. No confidence parameter softens HARD limits.  
2. Every breach has `reason`, `observed`, `threshold`, `severity`, `scope`.  
3. Live trading must call `validate_position` before execution (architectural rule 9).
