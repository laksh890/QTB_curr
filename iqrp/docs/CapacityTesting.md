# Capacity Testing

Capital → return / Sharpe / cost / drawdown curves and capacity limits.

---

## Purpose

Capacity testing estimates how strategy quality degrades as AUM grows under an impact/cost model. It answers: **at what capital do Sharpe and drawdown gates still pass?** Capacity is a scorecard dimension and a promotion gate — not a vanity AUM number.

**Package:** `iqrp.app.backtesting.capacity`  
**Primary functions:** `capacity_curve`, `estimate_capacity_limit`  
**Related:** [PerformanceMetrics](PerformanceMetrics.md) · [StrategyValidation](StrategyValidation.md) · [ParameterRobustness](ParameterRobustness.md)

---

## Architecture

```text
base causal returns
        │
CapacityModel (ADV, turnover, impact_coef/exp, fixed_cost)
        │
per capital level: adjust_returns = r - per_period_cost(capital)
        │
        ├── expected_return
        ├── expected_sharpe
        ├── expected_cost
        └── expected_drawdown
                │
        estimate_capacity_limit (max capital passing gates)
```

Default drag model:

```text
participation = capital / ADV
impact = impact_coef * participation ** impact_exp
annual_cost ≈ fixed_cost + impact * turnover
per_period_cost = annual_cost / periods_per_year
```

---

## Key APIs

### `CapacityModel`

```python
from iqrp.app.backtesting.capacity import CapacityModel

model = CapacityModel(
    adv=1e8,
    turnover=1.0,
    impact_coef=0.1,
    impact_exp=0.5,
    fixed_cost=0.0,
    periods_per_year=252.0,
)
drag = model.per_period_cost(capital=5e7)
adj = model.adjust_returns(rets, capital=5e7)
```

### `capacity_curve`

```python
import numpy as np
from iqrp.app.backtesting.capacity import capacity_curve

curve = capacity_curve(
    rets,
    capital_levels=np.geomspace(1e6, 1e9, 12),
    model=model,
)
# curve["capital"], ["expected_return"], ["expected_sharpe"],
# ["expected_cost"], ["expected_drawdown"], ["n_levels"]
```

Optional `cost_fn(capital) -> per_period_drag` overrides the model.

### `estimate_capacity_limit`

```python
from iqrp.app.backtesting.capacity import estimate_capacity_limit

limit = estimate_capacity_limit(
    rets,
    capital_levels=np.geomspace(1e6, 1e9, 16),
    min_sharpe=0.5,
    max_drawdown=0.25,
    model=model,
)
# limit["capacity_limit"] — largest capital still passing gates (0 if none)
```

### Via `BacktestEngine`

```python
result = bt.run(returns=rets, signals=sigs)
cap = bt.capacity_test(result=result, min_sharpe=0.5, max_drawdown=0.3)
print(cap["limit"]["capacity_limit"])
```

Default levels: `np.geomspace(1e6, 1e9, 12)`.

---

## Critical rules

| Rule | Detail |
|------|--------|
| Use causal base returns | Capacity on look-ahead paths is meaningless |
| Gates define “limit” | Limit is the max capital where Sharpe ≥ min and DD ≤ max |
| Costs scale with size | Ignoring impact overstates capacity |
| Capacity ≠ promotion alone | Still need OOS and full gate suite |
| Curve length | Meaningful curves need multiple capital levels |

---

## Integration

- Feed `capacity` into `build_scorecard(..., capacity=limit)` and `GateThresholds.min_capacity`
- Impact coefficients may be calibrated using Execution TCA imports; Execution package is not modified
- Portfolio ADV / participation assumptions are caller-supplied

---

## Example: attach capacity to scorecard

```python
from iqrp.app.backtesting.capacity import estimate_capacity_limit
from iqrp.app.backtesting.performance import build_scorecard

lim = estimate_capacity_limit(rets, min_sharpe=0.5, max_drawdown=0.25)
sc = build_scorecard(rets, oos_returns=oos, capacity=lim["capacity_limit"])
```
