# Backtesting Platform

Institutional Backtesting Platform (`iqrp.app.backtesting`).

## Critical rules

- No event handler may access data after the event timestamp (PIT).
- Look-ahead / leakage / invalid universe → INVALIDATED.
- Every run records data/feature/model/risk/portfolio/execution/code versions + seed.
- Never promote on highest historical return or Sharpe alone.

## Orchestrator

`BacktestEngine` runs the full pipeline, walk-forward, rolling retrain, scenarios, capacity, sweeps, ablation, comparison, scorecard, promotion gates, and paper-trading handoff.

---

## Purpose

The Institutional Backtesting Platform is the **Phase 13** research-to-promotion layer for IQRP. It evaluates strategies under point-in-time (PIT) constraints, measures multi-dimensional performance, stress-tests capacity and scenarios, and enforces promotion gates before paper trading.

It does **not** invent alpha, portfolio targets, or live orders. Upstream platforms supply signals/weights; this package simulates causality, costs, and evidence quality.

**Package:** `iqrp.app.backtesting`  
**Primary type:** `BacktestEngine`  
**Hydra config:** `iqrp/configs/backtesting/default.yaml`  
**Phase summary:** [Phase13_BacktestingPlatform.md](Phase13_BacktestingPlatform.md)

Related: [EventEngine](EventEngine.md) · [WalkForward](WalkForward.md) · [RollingRetraining](RollingRetraining.md) · [PerformanceMetrics](PerformanceMetrics.md) · [ScenarioTesting](ScenarioTesting.md) · [CapacityTesting](CapacityTesting.md) · [ParameterRobustness](ParameterRobustness.md) · [Reproducibility](Reproducibility.md) · [StrategyValidation](StrategyValidation.md)

> **Phase numbering:** Execution is **Phase 12**. This backtesting platform is **Phase 13**.

---

## Placement

```text
Alpha / Signals / Forecasts
        │
        ▼
Portfolio Construction (targets) ── imports only ──► Risk Intelligence
        │
        ▼
   BacktestEngine (Phase 13)
     PIT validate → simulate → metrics → WF / retrain / scenarios / capacity
        │
        ▼
   Scorecard + Gates ──► PaperTradingConfig (versions preserved)
        │
        ▼
   Execution (Phase 12) — optional TCA import for cost estimates
```

Integration with Execution, Risk, Portfolio, and Simulation is **import-only**. No modules outside `iqrp/app/backtesting/` were modified for Phase 13. Optional Execution TCA (`pre_trade_cost_estimate`) is used when available; otherwise the engine falls back to bps costs from settings.

---

## Package architecture

```text
iqrp.app.backtesting
├── engine.py                 BacktestEngine / BacktestResult
├── config.py                 BacktestSettings (Hydra / pydantic)
├── types.py                  BacktestState lifecycle
├── pit.py                    Look-ahead / universe / leakage
├── corporate_actions.py      Splits, dividends, mergers (PIT-filtered)
├── event_engine/             Deterministic MARKET→…→SETTLEMENT loop
├── walk_forward/             Rolling / expanding / purged / embargo
├── rolling_retraining/       Schedule-driven versioned retrains
├── performance/              Metrics + StrategyScorecard
├── scenarios/                Historical / MC / regime / liquidity / …
├── capacity.py               Capital → Sharpe / cost / DD curves
├── robustness.py             Sweeps, sensitivity, ablation
├── comparison.py             Multi-strategy comparison
├── validation_gates.py       OOS-mandatory promotion gates
├── paper_trading.py          Version-preserving handoff
├── experiment_registry.py    Lineage + rejected-run retention
├── serializer.py             JSON persistence
├── reports.py                Aggregate reporting
├── registry.py               Component registry
└── phase13.py                Completion validator
```

---

## Lifecycle states

`BacktestState` (`iqrp.app.backtesting.types`):

| State | Meaning |
|-------|---------|
| `CREATED` | Experiment registered |
| `VALIDATING` | PIT / leakage / universe checks |
| `RUNNING` | Causal simulation |
| `COMPLETED` | Successful terminal state |
| `FAILED` | Exception during run |
| `INVALIDATED` | Look-ahead, leakage, or invalid universe |
| `ARCHIVED` | Retained historical record |

**Rule:** leakage / look-ahead / invalid universe must transition to `INVALIDATED`, never silently continue.

---

## Key APIs

### `BacktestEngine`

```python
from iqrp.app.backtesting import BacktestEngine, BacktestSettings

engine = BacktestEngine(settings=BacktestSettings.default())
```

| Method | Role |
|--------|------|
| `run(...)` | PIT pipeline: validate → simulate → metrics → scorecard |
| `walk_forward(...)` | Causal folds via `WalkForwardEngine` |
| `retrain_rolling(...)` | Schedule-driven `RollingRetrainer` |
| `scenarios(...)` | `ScenarioEngine` dispatch |
| `capacity_test(...)` | Capacity curve + limit |
| `parameter_sweep` / `ablation` / `sensitivity` | Robustness |
| `compare(...)` | Multi-strategy comparison |
| `scorecard(...)` | Build / return `StrategyScorecard` |
| `validate_for_promotion(...)` | OOS-mandatory gates |
| `to_paper_trading(...)` | Preserve lineage for paper |
| `invalidate` / `save` / `load` | Audit + persistence |

### `BacktestResult`

Fields include `experiment_id`, `state`, `returns`, `equity`, `metrics`, `trades`, `exposures`, `costs`, `attribution`, `lineage` (`ExperimentLineage`), `seed`, `config`, `warnings`, `invalidated`, `invalidation_reason`, `oos_returns`, `scorecard`, `timestamps`. Serializes via `to_dict()` / `from_dict()`.

### `BacktestSettings`

Frozen pydantic settings loaded from Hydra (`configs/backtesting/default.yaml`):

- `clock`, `event_engine`, `pit`, `corporate_actions`
- `costs`, `latency`, `walk_forward`
- `reproducibility` (versions + seed)
- `reporting`, `initial_cash`, `name`

```python
settings = BacktestSettings.from_hydra(overrides=["costs.commission_bps=1.0"])
```

### Public package exports

`BacktestEngine`, `BacktestResult`, `BacktestSettings`, `BacktestState`, `ExperimentLineage`, `ExperimentRegistry`, `GateResult`, `GateThresholds`, `PaperTradingConfig`, `PaperTradingInterface`, `StrategyScorecard`, `build_scorecard`, `evaluate_gates`, `sharpe_ratio`, `ScenarioEngine`, `HistoricalScenario`, `capacity_curve`, `estimate_capacity_limit`, `compare_strategies`, `parameter_sweep`, `ablation_test`.

---

## Causal simulation (`run`)

1. Create experiment in `ExperimentRegistry` with full lineage from settings.
2. Enter `VALIDATING`:
   - Optional `filter_universe_asof` (survivorship guard).
   - Optional `detect_leakage` on feature/label as-of indices.
3. Enter `RUNNING` and simulate bar-by-bar:
   - Strategy/signal sees only `history = returns[:t+1]` (PIT).
   - PnL uses **previous** weight × current bar return, minus turnover costs.
   - Corporate dividends applied only via `actions_asof` when timestamps are aware.
4. Build metrics + `StrategyScorecard` (optional OOS slice via `oos_fraction`).
5. Register result; return `BacktestResult`.

```python
import numpy as np
from iqrp.app.backtesting import BacktestEngine

eng = BacktestEngine()
prices = 100 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 500))

def strategy(t, history):
    # Momentum on past only — history ends at t
    if len(history) < 20:
        return 0.0
    return float(np.sign(np.mean(history[-20:])))

result = eng.run(prices=prices, strategy_fn=strategy, seed=42, oos_fraction=0.2)
assert result.state.value == "COMPLETED"
gate = eng.validate_for_promotion(result)
# High IS Sharpe alone never approves without OOS
```

Invalidation path:

```python
# Feature as-of behind label as-of → INVALIDATED
result = eng.run(
    returns=np.random.randn(50) * 0.01,
    feature_asof_index=[0, 1, 2],
    label_asof_index=[0, 1, 3],
)
# result.state == INVALIDATED when leakage enabled in settings
```

---

## Corporate actions

`iqrp.app.backtesting.corporate_actions`:

- Types: `SPLIT`, `DIVIDEND`, `MERGER`, `DELISTING`, `SYMBOL_CHANGE`
- `actions_asof(actions, asof)` — only `ex_date <= asof`
- Split helpers adjust price/quantity; dividends can boost period PnL in the engine when enabled

Future corporate actions are look-ahead and must never be applied.

---

## Component registry and reports

- `registry.default_registry` — named component lookup for tooling
- `reports.full_report` — aggregate markdown/JSON style summaries from results

---

## Critical rules (platform)

| # | Rule |
|---|------|
| 1 | **PIT only.** No handler or strategy may read data after event/clock time. |
| 2 | **Invalidate on leakage.** Look-ahead, leakage, invalid universe → `INVALIDATED`. |
| 3 | **Full lineage.** Data/feature/label/model/risk/portfolio/execution/code versions + seed on every run. |
| 4 | **OOS mandatory for promotion.** Historical Sharpe/return alone never promotes. |
| 5 | **Rejected runs retained.** Registry keeps failed/invalidated experiments. |
| 6 | **Paper preserves versions.** Handoff carries strategy/feature/model/execution config. |
| 7 | **Import-only integration.** Execution TCA / risk metrics optional; no mutation of those packages. |
| 8 | **Costs are first-class.** Commission, spread, slippage (and optional TCA) enter PnL. |

---

## Integration notes

| Platform | How backtesting uses it |
|----------|-------------------------|
| Execution (Phase 12) | Optional `pre_trade_cost_estimate` inside cost simulation |
| Risk | Metrics/scorecard risk dimensions; no Risk package edits |
| Portfolio | Caller-supplied weights/signals; no Portfolio package edits |
| Simulation | Scenario/MC paths for stress; import-only |

Validate Phase 13:

```python
from iqrp.app.backtesting.phase13 import validate_phase13, write_phase13_report

report = validate_phase13()
write_phase13_report()  # docs/Phase13_BacktestingPlatform_Validation.json
```

---

## Operational runner and data pipeline

Phase 13 also ships an **executable operational layer** for user-supplied historical datasets (still Phase 13; Execution remains Phase 12):

| Concern | Entry | Docs |
|---------|-------|------|
| Runner lifecycle | `iqrp.app.backtesting.runner.BacktestRunner` | [BacktestRunner](BacktestRunner.md) |
| CLI | `python -m iqrp.app.backtesting.run` | [UserGuide](UserGuide.md) |
| Run config | `BacktestRunConfig` | [BacktestConfiguration](BacktestConfiguration.md) |
| Data ingest / registry | `iqrp.app.backtesting.data` | [DataPipeline](DataPipeline.md) · [DataAdapters](DataAdapters.md) |
| Validation / PIT | `DatasetValidator`, `point_in_time` | [DatasetValidation](DatasetValidation.md) · [PointInTimeData](PointInTimeData.md) |
| Cascade execution | `PipelineExecutor` + `EventPipeline` | [BacktestExecution](BacktestExecution.md) |
| Artifacts / reports | `persist_result`, `write_reports` | [ResultStorage](ResultStorage.md) · [BacktestReports](BacktestReports.md) |
| Reproducibility | config + checksum + seed | [Reproducibility](Reproducibility.md) |

`BacktestEngine` remains the research/promotion orchestrator. `BacktestRunner` drives a production-equivalent MARKET→…→RISK_UPDATE path over local Parquet/CSV with accounting ledgers. Neither path downloads market data; reference YAML such as `example_nifty50.yaml` is wiring only and is not a profitability claim.
