# Backtest Execution

How the operational Phase 13 runner turns chronological bars into a production-equivalent event cascade with accounting and optional research extensions.

---

## Purpose

Document the executable path: `PipelineExecutor` → `EventDrivenEngine` + `EventPipeline` → portfolio/execution adapters → ledgers → integrity → persistence.

**Modules:** `runner/executor.py`, `runner/pipeline.py`, `runner/adapters.py`, `runner/context.py`, `accounting/*`  
**CLI:** `python -m iqrp.app.backtesting.run`  
**Related:** [BacktestRunner](BacktestRunner.md) · [EventEngine](EventEngine.md) · [PointInTimeData](PointInTimeData.md) · [ResultStorage](ResultStorage.md)

> Execution **platform** (live/sim order routing) is **Phase 12**. This document covers Phase 13 **backtest** execution of the simulated cascade.

---

## Execution flow

```text
1. validate()     load + DatasetValidator + preflight
2. prepare()      StrategyRegistry.create → PipelineExecutor.prepare
                  CapitalState, PositionBook, BacktestClock, EventPipeline
3. run()          submit MarketEvent per timestamp → engine.run(start, end)
4. cascade/bar    MARKET→FEATURE→SIGNAL→FORECAST→RISK→PORTFOLIO
                  →ORDER→EXECUTION→FILL→POSITION→PNL→RISK_UPDATE
5. on_end         strategy hook
6. persist        results/{backtest_id}/…
7. integrity      reconcile_capital + ValidationReport
8. reports        report.md / report.json
```

`bars_by_timestamp` groups OHLCV into `(tz-aware timestamp → {instrument: bar})`. Resume skips bars with `ts <= resume_after`.

---

## Event cascade (operational handlers)

`EventPipeline` registers handlers that:

| Stage | Behavior |
|-------|----------|
| MARKET | PIT check, update prices, strategy `on_market_data` / `on_bar` |
| FEATURE | Simple return features + `on_features` |
| SIGNAL | Merge signals / targets; portfolio adapter may map signals → weights |
| FORECAST | Pass-through + `on_forecast` |
| RISK | Gross leverage clamp (`risk_config.max_gross_leverage`); risk_state |
| PORTFOLIO | Universe filter; `plan_from_targets` → pending orders |
| ORDER | Log orders; cost estimate |
| EXECUTION | `simulate_execution` (seeded); emit FILL events |
| FILL | Update fills, positions, cash, fees, trades |
| POSITION / PNL | Mark-to-market, equity/returns, portfolio snapshots |
| RISK_UPDATE | Update risk_state; `max_drawdown` breach → invalidate |

Priorities follow the Event Engine taxonomy (MARKET earliest within a timestamp). See [EventEngine](EventEngine.md).

---

## Adapters

| Adapter | Production import | Fallback |
|---------|-------------------|----------|
| `PortfolioConstructionAdapter` | `iqrp.app.portfolio` | `IsolatedPortfolioFallback` |
| `ExecutionSimulationAdapter` | Execution TCA / sim helpers when available | Isolated deterministic sim |

`backend` names are recorded on context diagnostics and in reports for audit.

---

## Accounting

Ledgers under `iqrp.app.backtesting.accounting`:

- `CapitalState` — cash, equity, realized/unrealized, fees, financing
- `PositionBook` — quantities / market values / exposure
- `OrderLog` / `FillLog` / `TradeLedger`
- `SnapshotBook` / `PortfolioSnapshot` — bar-level risk/PnL snapshot
- `reconcile_capital` — identity:  
  `Starting + Realized + Unrealized − Fees − Financing = Ending Equity`  
  within `reconciliation_tolerance`

Critical reconciliation failures fail the run.

---

## Pause / resume / cancel

- `pause()` sets `pause_requested` and writes `checkpoint.json` under `checkpoint_dir` or `output_dir` / `{backtest_id}/`.
- `resume()` / `--resume` restores context via `restore_context` and continues after last `current_time`.
- `cancel()` sets `cancel_requested` and transitions to `CANCELLED`.

---

## Parallel sweeps

```python
runner.parameter_sweep([{"strategy_params": {"lookback": 10}}, {"strategy_params": {"lookback": 20}}])
```

Or CLI `--parallel` when `parallel.grid` is set in config. Workers are process-isolated with unique `backtest_id` / seed (`runner/parallel.py`).

---

## Optional research modes

When configured on the run config (not enabled by default):

- `walk_forward_config` → `WalkForwardEngine` window summary attached to result
- `scenario_config` → `ScenarioEngine` stub summary
- `model_config.enabled` → rolling retrain hook metadata

These reuse existing Phase 13 research engines; they do not redesign them.

---

## Critical rules

| Rule | Detail |
|------|--------|
| Chronological bars only | No future bar access |
| PIT on MARKET | Violation → `INVALIDATED` |
| Seeded execution sim | Pass `seed` for reproducibility |
| Costs enter cash/fees | Commission/spread/slippage bps |
| Hard risk breach | e.g. drawdown > max → invalidate |
| No fabricated fills | Simulation only; model-based approximations disclosed in reports |
