# Parameter Robustness

Parameter sweeps, sensitivity analysis, ablation, stability regions, overfitting risk diagnostics.

---

## Purpose

Robustness tools measure whether edge survives nearby parameters and component removal. A sharp peak at a single grid point with large IS→OOS degradation is a red flag — not a promotion signal.

**Package:** `iqrp.app.backtesting.robustness`  
**Primary functions:** `parameter_sweep`, `sensitivity_analysis`, `ablation_test`, `stability_regions`, `overfitting_risk`  
**Related:** [WalkForward](WalkForward.md) · [PerformanceMetrics](PerformanceMetrics.md) · [StrategyValidation](StrategyValidation.md)

---

## Architecture

```text
objective(**params) -> returns | {"returns": ...}
        │
        ├── parameter_sweep      full grid → metrics surface + best
        ├── sensitivity_analysis one-at-a-time scales around base
        ├── ablation_test        toggle components off vs all-on
        ├── stability_regions    subset of sweep passing Sharpe/DD
        └── overfitting_risk     IS vs OOS Sharpe degradation score
```

---

## Key APIs

### `parameter_sweep`

```python
from iqrp.app.backtesting.robustness import parameter_sweep

def objective(lookback=20, thresh=0.0):
    # Build causal returns for these params
    return sim_returns(lookback=lookback, thresh=thresh)

sweep = parameter_sweep(
    objective,
    {"lookback": [10, 20, 40], "thresh": [0.0, 0.01]},
    periods_per_year=252.0,
)
# sweep["results"], ["surface"], ["best"], ["n_combinations"]
```

`objective` must return a return series or a dict containing `returns`.

### `sensitivity_analysis`

```python
from iqrp.app.backtesting.robustness import sensitivity_analysis

sens = sensitivity_analysis(
    objective,
    base_params={"lookback": 20, "thresh": 0.0},
    scales=(0.8, 0.9, 1.0, 1.1, 1.2),
)
# per-key curves + sharpe_range
```

Numeric knobs only; non-numeric params are skipped.

### `ablation_test`

```python
from iqrp.app.backtesting.robustness import ablation_test

abl = ablation_test(
    objective,
    components={"momentum": True, "mean_rev": True, "costs": True},
    base_params={"lookback": 20},
)
# results include ablation="none" baseline and per-component delta_sharpe
```

`objective` receives `base_params` plus boolean flags. Typical ablations: features, signals, models, risk overlays, execution, costs.

### `stability_regions`

```python
from iqrp.app.backtesting.robustness import stability_regions

stable = stability_regions(sweep, min_sharpe=0.5, max_drawdown=0.3)
# fraction of grid inside the region
```

### `overfitting_risk`

```python
from iqrp.app.backtesting.robustness import overfitting_risk

risk = overfitting_risk(in_sample_rets, oos_rets)
# in_sample_sharpe, out_of_sample_sharpe, degradation, risk_score ∈ [0, 1]
```

Diagnostics only — not a green light for promotion.

### Via `BacktestEngine`

```python
bt.parameter_sweep(objective, param_grid)
bt.ablation(objective, components={...})
bt.sensitivity(objective, base_params={...})
```

---

## Critical rules

| Rule | Detail |
|------|--------|
| Causal objectives | Sweep callbacks must not use future data |
| Prefer plateaus | Wide stability regions beat single-point optima |
| IS≠OOS | Use `overfitting_risk` + walk-forward; never promote on IS peak Sharpe |
| Ablate costs/risk | Removing costs often inflates Sharpe — treat as diagnostic |
| Best-of-sweep ≠ approved | `sweep["best"]` is exploratory, not a promotion decision |

---

## Integration

- Pair with [WalkForward](WalkForward.md) OOS folds inside `objective`
- Feed stability / degradation evidence into gate reviews and scorecard metadata
- Hyperparameter optimizers from other packages may supply grids; robustness stays in backtesting

---

## Example: sweep then stability

```python
sweep = parameter_sweep(objective, {"lb": [10, 20, 30]})
reg = stability_regions(sweep, min_sharpe=0.3, max_drawdown=0.4)
print(reg["fraction"], reg["n_stable"])
```
