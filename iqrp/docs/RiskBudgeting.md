# Risk Budgeting

Hierarchical risk budgets across scopes and risk types, with correlation-aware effective budgets for crowded sleeves.

Package: `iqrp.app.risk.capital`  
Builders: `build_risk_budgets`, `effective_risk_budgets`, `optimize_risk_budgets`  
Types: `RiskBudget`

Related: [Capital Allocation](CapitalAllocation.md) · [Risk Framework](RiskFramework.md) · [VaR](VaR.md) · [Expected Shortfall](ExpectedShortfall.md)

---

## Purpose

A **risk budget** is an explicit allowance of risk consumption at a named scope and risk type. Capital allocation converts budgets into weights; portfolio risk monitoring marks usage. Confidence never expands a budget beyond its configured limit.

---

## Hierarchical scopes

Defined in `iqrp.app.risk.capital.risk_budget.SCOPES`:

| Scope | Typical use |
|-------|-------------|
| `portfolio` | Aggregate book risk ceiling |
| `strategy` | Per-sleeve primary budgets (default construction) |
| `asset` | Single-name risk caps |
| `sector` | Industry / sector sleeves |
| `factor` | Style / systematic factor budgets |
| `market` | Market / beta / country buckets |
| `account` | Legal-entity or prime-broker account limits |

`build_risk_budgets` always emits:

1. One **strategy / volatility** budget per name (equal share of `total_risk_budget` unless `risk_budgets` overrides).
2. One **portfolio / volatility** aggregate named `"portfolio"`.
3. Optional entries from `scopes` and `risk_types` maps.

```python
from iqrp.app.risk.capital import CapitalAllocator, build_risk_budgets

budgets = build_risk_budgets(
    ["mom", "mr", "carry"],
    risk_budgets={"mom": 0.4, "mr": 0.35, "carry": 0.25},
    scopes={
        "sector": {"tech": 0.3, "rates": 0.4},
        "factor": {"value": 0.25, "momentum": 0.35},
        "account": {"prime_a": 1.0},
    },
    risk_types={
        "var": 0.05,
        "cvar": 0.08,
        "liquidity": 0.20,
        "drawdown": 0.15,
    },
)

# Via allocator
alloc = CapitalAllocator()
budgets = alloc.risk_budget(
    ["mom", "mr"],
    scopes={"market": {"equity": 0.6}},
    risk_types={"tail": 0.10},
)
```

`scopes` / `risk_types` values may be a single float (one budget for that scope/type) or `{name: budget}` maps.

---

## Risk types

Defined in `RISK_TYPES`:

| Risk type | Meaning |
|-----------|---------|
| `volatility` | Vol / variance contribution (primary capital budgets) |
| `var` | Value-at-Risk allowance |
| `cvar` | Conditional VaR / expected shortfall allowance |
| `liquidity` | Liquidity consumption budget |
| `concentration` | Concentration / HHI budget |
| `drawdown` | Drawdown depth allowance |
| `factor` | Factor exposure risk budget |
| `tail` | Tail / extreme-loss budget |

Unknown scope or risk-type keys are coerced to `portfolio` / `volatility` respectively so invalid config cannot silently invent open-ended budgets.

---

## `RiskBudget` fields

```python
from iqrp.app.risk.capital import RiskBudget

b = RiskBudget(
    name="mom",
    scope="strategy",
    risk_type="volatility",
    budget=0.35,
)
b.used = 0.10
b.remaining()  # 0.25
b.to_dict()    # full audit payload
```

| Field | Role |
|-------|------|
| `name` | Entity within the scope |
| `scope` | Hierarchical level |
| `risk_type` | Risk dimension of the budget |
| `budget` | Nominal allowance |
| `used` | Consumed amount after allocation (`mark_budgets_used`) |
| `confidence` | Clipped `[0, 1]` metadata — does not raise `budget` |
| `data_version` / `model_version` | Reproducibility |
| `reasons` | Construction provenance |

---

## Optimization objectives

`optimize_risk_budgets` (used by `CapitalAllocator.optimize`) supports:

| Objective | Behavior |
|-----------|----------|
| `min_risk` | ERC / minimum-risk seed |
| `max_diversification` | Inverse-volatility diversification |
| `target_volatility` | Scale toward portfolio vol target under leverage cap |
| `target_cvar` | Left-tail / CVaR-aware shrink |
| `target_drawdown` | Drawdown-aware conservative weights |
| `risk_budget_match` | Iterative match of risk contributions to target budgets (default) |
| `max_risk_adjusted_opportunity` | Opportunity tilt **inside** hard weight / leverage caps |

Hard constraints (`max_weight`, `max_leverage`, concentration via projection) always bind after optimization. Optimized budgets are then passed through the full `allocate` pipeline — optimization never bypasses capacity, correlation crowding, or risk-state zeros.

```python
from iqrp.app.risk.capital import CapitalAllocator
import numpy as np

alloc = CapitalAllocator()
result = alloc.optimize(
    ["a", "b", "c"],
    objective="target_volatility",
    cov=np.eye(3) * 0.015**2,
    target_vol=0.10,
    capital=1e7,
)
```

---

## Effective risk budgets under correlation crowding

Nominal budgets assume independent risk consumption. When strategies co-move, treating each nominal budget as additive overstates capacity. Capital allocation therefore computes **effective** budgets:

```python
from iqrp.app.risk.capital import (
    correlation_crowding_scales,
    effective_risk_budgets,
    strategy_correlation,
)

corr = strategy_correlation(returns)["matrix"]
eff = effective_risk_budgets(
    {"mom": 0.4, "mr": 0.35, "carry": 0.25},
    corr,
    names=["mom", "mr", "carry"],
    threshold=0.60,  # CapitalSettings.correlation_crowding_threshold
    floor=0.25,      # CapitalSettings.correlation_scale_floor
)
# eff["nominal"], eff["scales"], eff["effective"]
```

### Crowding scale

For name \(i\):

\[
\text{scale}_i = \mathrm{clip}\!\left(\frac{1}{1 + \sum_{j \neq i}\max(0,\,\rho_{ij}-\tau)},\; f,\; 1\right)
\]

where \(\tau\) is `correlation_crowding_threshold` and \(f\) is `correlation_scale_floor`.

Effective budget: \(\text{budget}_i \times \text{scale}_i\), then **renormalized** to preserve total budget mass when possible. Correlated sleeves therefore **share** the book’s risk allowance rather than stacking independent budgets.

Also available:

- `factor_correlation` / `return_correlation` / `drawdown_correlation`
- `tail_dependence_matrix` via `iqrp.app.risk.tail.tail_dependence`

---

## Usage marking

After allocation, strategy-level usage is marked proportional to final weights × total effective budget mass:

```python
from iqrp.app.risk.capital.risk_budget import mark_budgets_used, strategy_budget_vector

vec = strategy_budget_vector(names, budgets)  # name → strategy/volatility budget
mark_budgets_used(budgets, used_by_name)
```

Portfolio-scope volatility budget `used` is the sum of strategy usage.

---

## Rules

1. **Budgets are hard ceilings** for allocation math; confidence metadata cannot raise them.
2. **Strategy / volatility** is the primary capital-allocation budget vector.
3. **Correlation crowding is mandatory** inside `CapitalAllocator.allocate` before method weights.
4. **Hierarchical scopes** are additive audit structures — they do not silently replace strategy budgets unless you pass them into optimization / custom pipelines.
5. **Opportunity and forecast confidence never authorize unlimited risk** — see [Capital Allocation](CapitalAllocation.md) architectural rules.
