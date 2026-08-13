# Capital Allocation

Institutional capital allocation across strategies under hard risk, capacity, correlation, and drawdown controls.

Package: `iqrp.app.risk.capital`  
Primary type: `CapitalAllocator`  
Hydra config: `iqrp/configs/risk/capital/default.yaml`

Related: [Risk Budgeting](RiskBudgeting.md) · [Capacity Management](CapacityManagement.md) · [Risk Framework](RiskFramework.md) · [Position Sizing](PositionSizing.md) · [Drawdown Control](DrawdownControl.md)

---

## Placement

```text
Risk measures / cov / ADV / drawdowns / risk state
                    │
                    ▼
            CapitalAllocator.allocate()
                    │
                    ▼
     weights · capital amounts · RiskBudget usage
                    │
                    ▼
     Portfolio construction / rebalance / execution sizing
```

Capital allocation **consumes** risk modules via imports and **never** invents alpha from historical mean returns alone.

---

## Architectural rules

| # | Rule |
|---|------|
| 1 | **Historical performance alone never sets weights.** Mean returns are not an allocation input. Optional `expected_opportunity` may tilt only when explicitly provided. |
| 2 | **Hard limits always bind.** `max_weight`, leverage, concentration, participation, and risk-state zeros cannot be softened by forecast confidence. |
| 3 | **Correlated strategies share effective risk budget.** Crowding above `correlation_crowding_threshold` downscales nominal budgets before weighting. |
| 4 | **Missing capacity / liquidity is conservative.** Absent ADV/spread never implies unlimited size; default scales apply. |
| 5 | **Confidence cannot expand capital.** Forecast confidence and model agreement are clipped to `[0, 1]` and only shrink confidence on the result; risk-state scales never exceed `1.0`. |
| 6 | **Full audit trail.** Every `CapitalAllocation` records inputs, params, constraints, adjustments, and reasons. |

---

## Quick start

```python
import numpy as np
from iqrp.app.risk.capital import CapitalAllocator, CapitalSettings

settings = CapitalSettings.default()
# Or: CapitalSettings.from_hydra(overrides=["method=hrp", "max_weight=0.30"])

alloc = CapitalAllocator(settings)
result = alloc.allocate(
    ["mom", "mr", "carry"],
    method="risk_parity",
    cov=np.eye(3) * 0.02**2,
    capital=10_000_000,
    adv=[5e6, 3e6, 2e6],
    spreads=[0.0005, 0.0008, 0.0012],
    risk_state="NORMAL",
)

result.weights          # {"mom": ..., "mr": ..., "carry": ...}
result.capital_amounts
result.constraints_applied
result.reasons
```

---

## `CapitalAllocator` API

### `allocate`

Full pipeline: resolve covariance → build risk budgets → correlation-aware effective budgets → method weights → optional opportunity tilt → capacity → drawdown / risk-state scale → correlation scale on weights → hard projection → participation clip → capital amounts + audit.

| Parameter | Role |
|-----------|------|
| `names` | Strategy (or sleeve) identifiers |
| `method` | Allocation method key (see below) |
| `cov` / `returns` / `vols` | Risk inputs; cov preferred, else sample cov from returns, else diagonal from vols |
| `risk_budgets` | Nominal strategy risk-budget map |
| `capital` | Total capital to distribute |
| `adv` / `spreads` | Liquidity for capacity |
| `drawdowns` | Per-name drawdown levels |
| `expected_opportunity` | Optional non-negative tilt (never from hist mean alone) |
| `forecast_confidence` / `model_agreement` | Audit confidence only (clipped) |
| `regime` / `risk_state` | Dynamic / portfolio ceiling scales |
| `scopes` / `risk_types` | Hierarchical `RiskBudget` construction |

Returns `CapitalAllocation`.

### `allocate_strategy`

Build a single `StrategyAllocation` for one name (capital, risk budget, hard caps from settings).

### `allocate_risk_budget`

Convenience wrapper: `allocate(..., method="risk_budget", ...)`.

### `allocate_capital`

Convenience wrapper: default method `equal_capital` with explicit `capital`.

### `risk_budget`

Return hierarchical `list[RiskBudget]` without running the full weight pipeline (`build_risk_budgets`).

### `capital_budget`

Map weights → notional capital amounts (`allocate_capital_budgets`).

### `capacity`

Standalone capacity / liquidity scales via `estimate_capacity` (see [Capacity Management](CapacityManagement.md)).

### `optimize`

Run `optimize_risk_budgets` under an objective, then feed optimized budgets into the full `allocate(..., method="risk_budget")` pipeline so hard limits still bind.

Objectives: `min_risk` · `max_diversification` · `target_volatility` · `target_cvar` · `target_drawdown` · `risk_budget_match` · `max_risk_adjusted_opportunity`.

### `allocate_scenarios`

Allocate across synthetic regimes (`independent`, `correlated`, `low_liquidity`, `high_volatility`, `regime`, `drawdown`, `tail`) via `simulate_capital_scenario`. Returns `dict[str, CapitalAllocation]`.

### `rebalance`

Move from current weights toward a target (`CapitalAllocation` or weight vector) under turnover and participation caps, then hard projection.

### `export_state`

Serialize settings + last allocation for audit / resume.

---

## Allocation methods

| Method key | Behavior |
|------------|----------|
| `equal_capital` | Equal weights `1/n` |
| `equal_risk` | Equal risk contribution (ERC) |
| `risk_parity` | Capital risk parity (default) |
| `risk_budget` | Risk-parity toward supplied / effective budgets |
| `volatility` | Volatility budgeting to `target_volatility` |
| `hrp` | Hierarchical Risk Parity |
| `herc` | Hierarchical Equal Risk Contribution |
| `correlation` | Risk parity seed × crowding scales |
| `drawdown` | Risk parity seed × drawdown state scales |
| `capacity` | Equal seed × capacity scales |
| `dynamic` | Multi-factor dynamic risk scales (regime, state, confidence, opportunity, liquidity) |

Unknown methods fall back to risk parity and record the reason.

---

## Pipeline stages (every `allocate` call)

1. **Covariance** — provided matrix, or `iqrp.app.risk.market.correlation.covariance_matrix` on returns, or diagonal vols, else small identity.
2. **Risk budgets** — `build_risk_budgets` with optional scopes / risk types.
3. **Effective budgets** — `effective_risk_budgets` so crowded names share budget mass.
4. **Method weights** — never from historical mean returns.
5. **Opportunity tilt** — only if `expected_opportunity` provided (skipped for `dynamic`).
6. **Capacity** — `estimate_capacity` (imports `liquidity_risk`).
7. **Drawdown + risk state** — per-name DD scales × portfolio `risk_state_scales` ceiling.
8. **Correlation on weights** — apply crowding scales again, renormalize.
9. **Hard projection** — `project_weights` (weight, gross, leverage, concentration).
10. **Participation** — ADV × TTL hard clip.
11. **Capital + usage** — amounts and `mark_budgets_used`.

---

## Result types

```python
from iqrp.app.risk.capital import CapitalAllocation, RiskBudget, StrategyAllocation
```

- **`CapitalAllocation`** — weights, capital amounts, risk budgets used, per-strategy allocations, correlation / capacity / drawdown adjustments, constraints, confidence, reasons, full `to_dict()` audit.
- **`RiskBudget`** — hierarchical scope × risk type with `budget`, `used`, `remaining()`.
- **`StrategyAllocation`** — capital/risk budgets plus `max_gross`, `max_net`, `max_position`, `max_leverage`, `max_turnover`, `max_participation`, and scale factors.

---

## Hydra configuration

Path: `iqrp/configs/risk/capital/default.yaml`

```yaml
method: risk_parity
max_weight: 0.40
max_gross_exposure: 1.5
max_leverage: 2.0
max_participation: 0.10
missing_capacity_scale: 0.50
missing_liquidity_scale: 0.50
correlation_crowding_threshold: 0.60
correlation_scale_floor: 0.25
target_volatility: 0.10
risk_state_scales:
  NORMAL: 1.0
  CAUTION: 0.8
  REDUCED_RISK: 0.5
  CAPITAL_PRESERVATION: 0.25
  TRADING_HALT: 0.0
```

Load:

```python
from iqrp.app.risk.capital import CapitalSettings

settings = CapitalSettings.from_hydra(
    overrides=["method=herc", "max_weight=0.25", "max_participation=0.05"]
)
```

---

## Integration with existing risk modules

Capital **imports** (does not reimplement) risk primitives:

| Concern | Import path |
|---------|-------------|
| Covariance / correlation | `iqrp.app.risk.market.correlation` |
| Liquidity / ADV / impact | `iqrp.app.risk.market.liquidity.liquidity_risk` |
| Drawdown series | `iqrp.app.risk.tail.drawdown` |
| Tail dependence | `iqrp.app.risk.tail.tail_dependence` |
| Portfolio vol | `iqrp.app.risk.portfolio.portfolio_risk` |
| Risk parity primitives | `iqrp.app.risk.sizing.risk_parity` |

```python
from iqrp.app.risk.capital import (
    CapitalAllocator,
    equal_risk_weights,
    capital_risk_parity,
    hrp_weights,
    herc_weights,
    volatility_budgets,
    effective_risk_budgets,
    estimate_capacity,
    optimize_risk_budgets,
)
```

---

## Rebalance example

```python
target = alloc.allocate(["mom", "mr", "carry"], method="hrp", cov=cov, capital=1e7)
current = {"mom": 0.5, "mr": 0.3, "carry": 0.2}
rb = alloc.rebalance(current, target, capital=1e7, adv=adv)
# constraints_applied may include turnover_cap, participation_cap
```
