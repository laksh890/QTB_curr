# Scenario Testing

Historical, hypothetical, Monte Carlo, regime, liquidity, volatility, correlation, and gap scenarios.

---

## Purpose

Scenario testing stresses a strategy path under user-defined history windows, shocks, stochastic paths, and regime/liquidity/vol/correlation/gap distortions. Results inform robustness and regime fields on the scorecard — they do **not** by themselves promote a strategy.

**Package:** `iqrp.app.backtesting.scenarios`  
**Primary type:** `ScenarioEngine`  
**Related:** [PerformanceMetrics](PerformanceMetrics.md) · [CapacityTesting](CapacityTesting.md) · [StrategyValidation](StrategyValidation.md)

---

## Architecture

```text
ScenarioEngine.run(kind, returns, ...)
        │
        ├── historical     user-defined window/mask (no hard-coded crises)
        ├── hypothetical   Shock list applied to path
        ├── monte_carlo    bootstrap / parametric / block / regime-aware
        ├── regime         per-regime metrics + evaluate_regime_robustness
        ├── liquidity      liquidity drought cost/impact stress
        ├── volatility     vol spike / scale scenarios
        ├── correlation    correlation breakdown / spike
        └── gap            overnight / jump gap shocks
```

Path-dependent `state` (positions, costs, drawdown) may be threaded through and echoed on the result.

---

## Key APIs

### `ScenarioEngine`

```python
from iqrp.app.backtesting.scenarios import ScenarioEngine, HistoricalScenario

eng = ScenarioEngine(n_simulations=500, seed=42, block_size=5)
```

| Method | Role |
|--------|------|
| `run(kind, returns, ...)` | Dispatch to a single scenario runner |
| `run_suite(returns, include=..., historical=...)` | Batch suite; continues on partial failure |

Kinds: `historical`, `hypothetical`, `monte_carlo`, `regime`, `liquidity`, `volatility`, `correlation`, `gap`.

### Historical (user-defined only)

```python
from iqrp.app.backtesting.scenarios import HistoricalScenario, run_historical_scenario

scenario = HistoricalScenario(name="custom_stress", start=100, end=150)
# Or mask=boolean_array — no embedded crisis calendars
out = run_historical_scenario(rets, scenario, periods_per_year=252.0)
```

`ScenarioEngine` raises if `kind="historical"` and `scenario` is missing.

### Hypothetical shocks

```python
from iqrp.app.backtesting.scenarios import HypotheticalShock, run_hypothetical_scenario

shocks = [HypotheticalShock(index=50, shock=-0.05), {"index": 80, "shock": -0.03}]
out = run_hypothetical_scenario(rets, shocks)
```

### Monte Carlo

```python
from iqrp.app.backtesting.scenarios import run_monte_carlo

out = run_monte_carlo(
    rets,
    method="bootstrap",  # engine also accepts method via ScenarioEngine
    n_simulations=500,
    seed=42,
    block_size=5,
)
```

### Regime

```python
from iqrp.app.backtesting.scenarios.regime import (
    classify_simple_regimes,
    run_regime_scenario,
    evaluate_regime_robustness,
)

labels = classify_simple_regimes(rets)
out = run_regime_scenario(rets, labels, regime="high_volatility")
robust = evaluate_regime_robustness(rets, labels)
```

Canonical label names include `trending`, `mean_reverting`, `high_volatility`, `low_volatility`, `high_correlation`, `low_correlation`, `low_liquidity`, `regime_transition`.

### Liquidity / volatility / correlation / gap

```python
from iqrp.app.backtesting.scenarios.liquidity import run_liquidity_scenario
from iqrp.app.backtesting.scenarios.volatility import run_volatility_scenario
from iqrp.app.backtesting.scenarios.correlation import run_correlation_scenario
from iqrp.app.backtesting.scenarios.gap import run_gap_scenario

run_liquidity_scenario(rets, severity=0.5)
run_volatility_scenario(rets, scale=2.0)
run_correlation_scenario(asset_rets_matrix, seed=42)
run_gap_scenario(rets, gap=-0.08, index=10)
```

### Via `BacktestEngine`

```python
bt.scenarios("monte_carlo", returns=rets, n_simulations=200)
bt.scenarios("historical", returns=rets, scenario=HistoricalScenario("w", start=0, end=60))
```

---

## Critical rules

| Rule | Detail |
|------|--------|
| No hard-coded crises | Historical windows are caller-supplied only |
| Scenarios ≠ promotion | Stress results inform robustness; gates still require OOS |
| Seeded MC | Monte Carlo uses explicit `seed` for reproducibility |
| PIT of base path | Scenario inputs should come from causal backtest returns |
| Suite isolation | `run_suite` records per-kind errors without aborting the rest |

---

## Integration

- Regime robustness can populate `StrategyScorecard.regime_robustness` via `build_scorecard(..., regime_returns=...)`
- Optional Simulation engine imports remain read-only
- Execution costs may be re-applied inside custom scenario callbacks; no Execution package edits

---

## Example: suite

```python
from iqrp.app.backtesting.scenarios import ScenarioEngine, HistoricalScenario

eng = ScenarioEngine(seed=0)
suite = eng.run_suite(
    rets,
    historical=HistoricalScenario("window", start=0, end=40),
    include=["historical", "volatility", "liquidity", "gap", "monte_carlo"],
)
print(suite["reports"].keys())
```
