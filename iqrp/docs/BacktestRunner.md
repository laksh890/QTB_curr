# BacktestRunner

Operational lifecycle for executable Phase 13 institutional backtests: create → validate → prepare → run (pause / resume / cancel) → persist → report.

> **Phase numbering:** Execution is **Phase 12**. This operational runner is part of **Phase 13** (Institutional Backtesting Platform). It does not replace `BacktestEngine`; it drives a production-equivalent event cascade over user-supplied historical data.

---

## Purpose

`BacktestRunner` turns a validated local dataset and an explicitly registered strategy into a complete simulated trading path: market bars → features → signals → risk → portfolio → orders → execution → fills → positions → PnL → risk update, with accounting ledgers, integrity checks, artifact persistence, and institutional reports.

**Package:** `iqrp.app.backtesting.runner`  
**Primary type:** `BacktestRunner`  
**CLI:** `python -m iqrp.app.backtesting.run`  
**Config:** `BacktestRunConfig` ([BacktestConfiguration](BacktestConfiguration.md))  
**Related:** [DataPipeline](DataPipeline.md) · [BacktestExecution](BacktestExecution.md) · [ResultStorage](ResultStorage.md) · [BacktestReports](BacktestReports.md) · [UserGuide](UserGuide.md) · [BacktestingPlatform](BacktestingPlatform.md) · [EventEngine](EventEngine.md)

---

## Architecture

```text
BacktestRunConfig (YAML / dict / OmegaConf)
        │
        ▼
   BacktestRunner
     ├── Lifecycle (CREATED → VALIDATING → PREPARING → RUNNING → …)
     ├── load_market_frame (ParquetAdapter / CSVAdapter + DatasetValidator)
     ├── StrategyRegistry.create(strategy_id, strategy_version, **params)
     ├── PipelineExecutor + EventPipeline (MARKET→…→RISK_UPDATE)
     ├── persist_result → results/{backtest_id}/…
     └── write_reports → report.md / report.json
```

Integration with Portfolio Construction and Execution is **import-only** via runner adapters (`PortfolioConstructionAdapter`, `ExecutionSimulationAdapter`) with named isolated fallbacks when production modules are unavailable.

---

## Lifecycle states

`RunnerLifecycleState` (`iqrp.app.backtesting.runner.lifecycle`):

| State | Meaning |
|-------|---------|
| `CREATED` | Runner constructed |
| `VALIDATING` | Preflight: strategy registered, dataset loads/validates, capital/dates |
| `PREPARING` | Strategy instantiated, bars loaded, `PipelineExecutor.prepare()` |
| `RUNNING` | Event cascade executing |
| `PAUSED` | Pause requested; checkpoint written |
| `COMPLETED` | Terminal success (may include soft integrity warnings) |
| `FAILED` | Preflight / integrity critical failure or exception |
| `CANCELLED` | Explicit cancel |
| `INVALIDATED` | Look-ahead / PIT breach or hard risk breach |
| `ARCHIVED` | Allowed transition from a terminal state |

Terminal states cannot transition except to `ARCHIVED`. History is timestamped (`Lifecycle.to_dict()`).

Mapping helpers exist between runner states and platform `BacktestState` (`map_engine_state` / `map_runner_to_engine`) without conflating the two machines.

---

## Key APIs

### Construction

```python
from iqrp.app.backtesting.runner import BacktestRunner, BacktestRunConfig

runner = BacktestRunner("iqrp/configs/backtesting/synthetic_demo.yaml")
# or
runner = BacktestRunner(BacktestRunConfig.from_yaml(...))
# or
runner = BacktestRunner({"strategy_id": "buy_and_hold", "dataset_path": "...", ...})
# optional strategy override (skips registry lookup for that instance)
runner = BacktestRunner(cfg, strategy=my_strategy)
```

### Lifecycle methods

| Method | Role |
|--------|------|
| `create()` | Explicit `CREATED` transition |
| `validate()` | Preflight; loads dataset when `dataset_path` set; raises `ValueError` if critical |
| `prepare()` | Resolve strategy, build `PipelineExecutor`, optional checkpoint restore |
| `run()` | Execute cascade, persist, integrity-validate, write reports |
| `pause()` | Request pause + write checkpoint under `checkpoint_dir` or `output_dir` |
| `resume()` | Restore from checkpoint path and continue `run()` |
| `cancel()` | Request cancel |
| `result()` | `OperationalBacktestResult` (after `run`, or built from context) |
| `report()` | Path to markdown (preferred) or JSON report |
| `status()` | Current `RunnerLifecycleState` |
| `parameter_sweep(grid)` | Process-isolated parallel sweeps |
| `walk_forward()` / `scenarios()` / `retrain()` | Optional research extensions when configured |

### CLI

```bash
.venv/bin/python -m iqrp.app.backtesting.run --config iqrp/configs/backtesting/synthetic_demo.yaml
```

Flags: `--strategy`, `--strategy-version`, `--dataset`, `--adapter {parquet,csv}`, `--start`, `--end`, `--capital`, `--universe`, `--output`, `--seed`, `--resume`, `--parallel`, `--backtest-id`. CLI registers reference strategies `buy_and_hold` and `cross_sectional_momentum` before run.

---

## Preflight and integrity

**Preflight** (`preflight_validate`): requires non-empty `strategy_id`, strategy registered (or override), `dataset_path` or `dataset_id`, successful dataset validation when path is resolvable, positive `initial_capital`, and `start <= end`.

**Integrity** (`integrity_validate`): data validated, PIT not invalidated, strategy ran (`bar_count > 0`), equity present, capital reconciliation via `reconcile_capital`, results persisted. Critical reconciliation / look-ahead failures fail or invalidate the run; missing persistence is a warning only.

Critical data-quality failures during `load_market_frame` raise before the cascade starts — there is no silent repair.

---

## Result object

`OperationalBacktestResult` carries `backtest_id`, `status`, equity/returns/timestamps, orders/fills/trades, positions log, portfolio snapshots, capital, performance, risk, execution, diagnostics, walk-forward/scenarios hooks, reconciliation, seed, and full config. Serialized via `to_dict()` / `from_dict()`.

Reports explicitly state that figures describe the simulated path under stated assumptions and are **not** a profitability claim.

---

## Strategies

Selection is always explicit via `strategy_id` (+ `strategy_version` when multiple versions exist). Built-in reference strategies (pipeline validation only):

| `strategy_id` | Class | Notes |
|---------------|-------|-------|
| `buy_and_hold` | `BuyAndHoldStrategy` | `equal_weight` or `first_instrument` |
| `cross_sectional_momentum` | `CrossSectionalMomentumStrategy` | lookback / top_n demo |

Custom strategies subclass `Strategy`, set `strategy_id` / `strategy_version`, and `StrategyRegistry.register(...)`.

---

## Critical rules

| Rule | Detail |
|------|--------|
| No remote market download | `dataset_path` must exist locally |
| Explicit strategy selection | Registry refuses silent / arbitrary pick |
| PIT enforced when `enforce_pit=True` | Look-ahead → `INVALIDATED` |
| Critical data issues fail the run | No silent repair |
| Same config + dataset checksum + seed | Expected deterministic path (see [Reproducibility](Reproducibility.md)) |
| Reference configs are not alpha | e.g. `example_nifty50.yaml` is wiring only |

---

## Example

```python
from iqrp.app.backtesting.runner import BacktestRunner
from iqrp.app.backtesting.strategy import BuyAndHoldStrategy, StrategyRegistry

StrategyRegistry.register(BuyAndHoldStrategy, overwrite=True)
runner = BacktestRunner("iqrp/configs/backtesting/synthetic_demo.yaml")
runner.validate()
runner.prepare()
result = runner.run()
assert runner.status().value == "COMPLETED"
print(runner.report())  # results/synthetic_demo/reports/report.md
```
