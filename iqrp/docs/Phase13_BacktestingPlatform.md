# Phase 13 — Institutional Backtesting Platform

**Status:** PASS

- Components passed: 41/41
- Docs present: 11/11

**Note:** Execution used Phase 12; this is Phase 13.

## Checklist

- [x] Event Engine
- [x] Event Queue
- [x] Deterministic Clock
- [x] Event-Driven Backtesting
- [x] Point-in-Time Validation
- [x] Survivorship-Bias Protection
- [x] Corporate Actions
- [x] Walk-Forward
- [x] Rolling Windows
- [x] Expanding Windows
- [x] Purged Validation
- [x] Embargo
- [x] Rolling Retraining
- [x] Model Versioning
- [x] Performance Metrics
- [x] Risk Metrics
- [x] Drawdown Metrics
- [x] Tail Metrics
- [x] Trade Metrics
- [x] Exposure Metrics
- [x] Performance Attribution
- [x] Benchmarking
- [x] Stability Analysis
- [x] Historical Scenarios
- [x] Hypothetical Scenarios
- [x] Monte Carlo Scenarios
- [x] Regime Scenarios
- [x] Liquidity Scenarios
- [x] Capacity Testing
- [x] Parameter Robustness
- [x] Ablation Testing
- [x] Strategy Comparison
- [x] Experiment Registry
- [x] Reproducibility
- [x] Strategy Validation Gates
- [x] Paper Trading Interface
- [x] Scorecard
- [x] Backtest Engine
- [x] Reports
- [x] Component Registry
- [x] Scenario Engine

---

## Purpose

Phase 13 delivers the Institutional Backtesting Platform: deterministic PIT simulation, walk-forward and rolling retrain OOS evidence, multi-metric scorecards, scenario/capacity/robustness stress, OOS-mandatory promotion gates, and version-preserving paper-trading handoff.

It sits **after** Alpha Research (Phase 11) and Execution (Phase 12) in the product numbering used by this repository.

---

## Architecture

```text
iqrp.app.backtesting
├── BacktestEngine          orchestrator (run / WF / retrain / scenarios / …)
├── event_engine/           clock + priority queue + handlers
├── walk_forward/           rolling / expanding / anchored / purged+embargo
├── rolling_retraining/     schedule triggers + versioned models
├── performance/            metrics + StrategyScorecard
├── scenarios/              historical / MC / regime / liquidity / …
├── capacity.py / robustness.py / comparison.py
├── validation_gates.py + paper_trading.py
├── experiment_registry.py + serializer.py + pit.py
└── phase13.py              import + docs + policy validator
```

Hydra config: `iqrp/configs/backtesting/default.yaml`.

Documentation map:

| Doc | Focus |
|-----|--------|
| [BacktestingPlatform.md](BacktestingPlatform.md) | Orchestrator overview |
| [EventEngine.md](EventEngine.md) | Deterministic event loop |
| [WalkForward.md](WalkForward.md) | Causal folds |
| [RollingRetraining.md](RollingRetraining.md) | Versioned refresh |
| [PerformanceMetrics.md](PerformanceMetrics.md) | Scorecard & ratios |
| [ScenarioTesting.md](ScenarioTesting.md) | Stress suite |
| [CapacityTesting.md](CapacityTesting.md) | AUM curves |
| [ParameterRobustness.md](ParameterRobustness.md) | Sweeps / ablation |
| [Reproducibility.md](Reproducibility.md) | Lineage / PIT / serialize |
| [StrategyValidation.md](StrategyValidation.md) | Gates + paper handoff |

---

## Architectural rules

- No event handler may access data after event.timestamp (PIT)
- Look-ahead / leakage / invalid universe → INVALIDATED
- Every run records versions + seed for reproducibility
- Never promote on highest historical return or Sharpe alone
- Out-of-sample evidence is mandatory for promotion
- Paper trading preserves strategy/feature/model/execution versions
- Rejected / invalidated experiments are retained in the registry
- Optional Execution TCA / risk imports — no hard dependency mutation

---

## Integration

- **No existing modules outside `iqrp/app/backtesting/` were modified.**
- Execution TCA (`pre_trade_cost_estimate`) and risk metrics are imported optionally when available; otherwise bps costs from settings are used.
- Portfolio / Risk / Simulation / Alpha packages are consumed by **import only** inside strategy callbacks and estimators.
- Hydra config: `iqrp/configs/backtesting/default.yaml`.
- Package exports (`iqrp.app.backtesting.__all__`) expose `BacktestEngine`, `BacktestSettings`, `BacktestResult`, `ExperimentRegistry`, gates, paper trading, scorecard helpers, scenarios, capacity, and robustness entry points.

### Phase numbering note

Product briefs sometimes confuse phase numbers. In this repository:

| Phase | Platform |
|-------|----------|
| 11 | Alpha Research |
| 12 | Institutional Execution |
| **13** | **Institutional Backtesting** |

---

## Validation

Run the machine-readable completion check:

```python
from iqrp.app.backtesting.phase13 import validate_phase13, write_phase13_report

report = validate_phase13()
assert report["status"] == "PASS"
write_phase13_report()  # writes Phase13_BacktestingPlatform_Validation.json
```

The validator confirms:

1. All Phase 13 components import and expose required symbols
2. Required docs exist under `iqrp/docs/`
3. `BacktestEngine` exposes the full orchestrator API
4. Gate policy: high IS Sharpe without OOS does **not** approve
5. Integration note: no external module mutation; optional TCA import

Machine-readable report: `Phase13_BacktestingPlatform_Validation.json`.

---

## Quick start

```python
import numpy as np
from iqrp.app.backtesting import BacktestEngine, GateThresholds

rets = np.random.default_rng(42).normal(0.0004, 0.01, 504)
eng = BacktestEngine()
result = eng.run(returns=rets, signals=np.sign(rets), oos_fraction=0.2, seed=42)
gate = eng.validate_for_promotion(result, GateThresholds())
if gate.approved:
    paper = eng.to_paper_trading(result)
```
