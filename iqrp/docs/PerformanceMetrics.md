# Performance Metrics

Returns, risk-adjusted, drawdown, tail, trade, exposure, attribution, stability, and StrategyScorecard.

---

## Purpose

The performance package turns causal backtest paths into institutional metrics and a multi-dimensional **scorecard**. Promotion decisions use the scorecard plus gates — never return or Sharpe alone.

**Package:** `iqrp.app.backtesting.performance`  
**Primary types:** `StrategyScorecard`, `build_scorecard`  
**Related:** [StrategyValidation](StrategyValidation.md) · [CapacityTesting](CapacityTesting.md) · [ScenarioTesting](ScenarioTesting.md) · [BacktestingPlatform](BacktestingPlatform.md)

---

## Architecture

```text
returns / positions / costs / oos_returns / regime_returns
        │
        ├── returns.py          total_return, cagr, summarize_returns
        ├── risk_adjusted.py    Sharpe, Sortino, Calmar, IR, captures
        ├── drawdown.py         max_drawdown, summarize_drawdown
        ├── tail.py             VaR / CVaR, summarize_tail
        ├── trade_metrics.py    turnover, win rate, summarize_trades
        ├── exposure.py         gross/net, summarize_exposure
        ├── attribution.py      full_attribution
        ├── benchmark.py        compare_to_benchmark
        ├── stability.py        rolling Sharpe stability_report
        └── scorecard.py        StrategyScorecard / build_scorecard
```

---

## Key APIs

### Returns

```python
from iqrp.app.backtesting.performance import (
    total_return, cagr, annualized_return, summarize_returns
)

summary = summarize_returns(rets)  # dict of return stats
```

### Risk-adjusted

```python
from iqrp.app.backtesting.performance import sharpe_ratio, sortino_ratio, summarize_risk_adjusted

sharpe_ratio(rets, risk_free=0.0, periods_per_year=252.0)
sortino_ratio(rets, mar=0.0, periods_per_year=252.0)
# Also: calmar_ratio, omega_ratio, information_ratio,
# upside_capture, downside_capture, capture_ratios
```

### Drawdown and tail

```python
from iqrp.app.backtesting.performance import max_drawdown, summarize_drawdown, summarize_tail

max_drawdown(rets)          # positive fraction peak-to-trough
summarize_tail(rets)        # includes CVaR etc.
```

### Trades, exposure, attribution, benchmark, stability

```python
from iqrp.app.backtesting.performance import (
    summarize_trades,
    summarize_exposure,
    full_attribution,
    compare_to_benchmark,
    stability_report,
)

stability_report(rets, window=63, periods_per_year=252.0)
compare_to_benchmark(strategy_rets, benchmark_rets)
full_attribution(...)  # strategy / factor-style breakdown when inputs provided
```

### Strategy comparison

```python
from iqrp.app.backtesting import compare_strategies

compare_strategies({"A": rets_a, "B": rets_b})
```

### `StrategyScorecard`

Fields:

| Field | Meaning |
|-------|---------|
| `total_return`, `cagr` | Wealth growth |
| `sharpe`, `sortino`, `calmar` | Risk-adjusted |
| `max_drawdown`, `cvar` | Path risk / tail |
| `turnover`, `transaction_costs` | Implementation burden |
| `capacity` | Optional capital capacity |
| `stability` | Rolling Sharpe mean / (1+std) |
| `regime_robustness` | Optional cross-regime score |
| `out_of_sample` | OOS Sharpe (required for promotion) |
| `metadata` | Extra audit fields |

```python
from iqrp.app.backtesting.performance import build_scorecard, StrategyScorecard

sc = build_scorecard(
    rets,
    positions=weights,
    costs=cost_series,
    oos_returns=oos,
    regime_returns={"high_vol": hv, "low_vol": lv},
    capacity=5e7,
)
assert isinstance(sc, StrategyScorecard)
gate_preview = sc.passes_gates(min_sharpe=0.5, max_drawdown=0.35, min_oos=0.0)
```

`passes_gates` is a helper on the scorecard; production promotion uses `evaluate_gates` in [StrategyValidation](StrategyValidation.md) (OOS mandatory).

### Via `BacktestEngine`

```python
result = engine.run(returns=rets, signals=sigs, oos_fraction=0.25)
sc = engine.scorecard(result)
```

---

## Critical rules

| Rule | Detail |
|------|--------|
| Multi-metric by design | Scorecard covers risk, costs, stability, capacity, OOS — not return/Sharpe alone |
| OOS is explicit | `out_of_sample=None` means “no OOS evidence,” not “perfect” |
| Costs in the scorecard | Transaction costs and turnover are first-class fields |
| Annualization consistency | Pass the same `periods_per_year` across ratios |
| Never promote on IS Sharpe | Even a Sharpe of 3.0 fails gates without OOS |

---

## Integration

- Consumed by validation gates and paper-trading handoff
- Optional Risk package metrics may be imported by callers; this package stays self-contained for core ratios
- Scenario / capacity modules reuse the same Sharpe / drawdown helpers for consistency

---

## Example: full scorecard from a run

```python
import numpy as np
from iqrp.app.backtesting import BacktestEngine

eng = BacktestEngine()
rets = np.random.default_rng(1).normal(0.0004, 0.01, 504)
result = eng.run(returns=rets, signals=np.tanh(rets), oos_fraction=0.2, costs=True)
sc = result.scorecard
print(sc.sharpe, sc.max_drawdown, sc.out_of_sample, sc.stability)
```
