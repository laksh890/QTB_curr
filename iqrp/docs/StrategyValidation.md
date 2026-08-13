# Strategy Validation

Promotion gates (OOS mandatory), paper-trading handoff preserving versions/configs. Never promote on hist Sharpe/return alone.

---

## Institutional Backtesting Platform (Phase 13)

This section documents promotion validation for the Phase 13 Institutional Backtesting Platform (`iqrp.app.backtesting`). Execution is Phase 12; backtesting validation and paper handoff are Phase 13.

### Purpose

Strategy validation converts a completed (non-invalidated) backtest into an explicit **approve / reject** decision using multi-dimensional gates, then optionally packages a **paper-trading handoff** that preserves strategy, feature, model, and execution versions.

**Modules:** `validation_gates`, `paper_trading`, `performance.scorecard`  
**Orchestrator entry points:** `BacktestEngine.validate_for_promotion`, `BacktestEngine.to_paper_trading`  
**Related:** [PerformanceMetrics](PerformanceMetrics.md) · [Reproducibility](Reproducibility.md) · [WalkForward](WalkForward.md) · [BacktestingPlatform](BacktestingPlatform.md) · [Phase13](Phase13_BacktestingPlatform.md)

### Architecture

```text
BacktestResult (COMPLETED, not INVALIDATED)
        │
        ▼
StrategyScorecard  (return, risk, DD, CVaR, turnover, costs,
                    stability, regime, capacity, OOS)
        │
        ▼
evaluate_gates / BacktestEngine.validate_for_promotion
        │
        ├── reject if OOS missing
        ├── reject if in-sample Sharpe alone
        ├── reject on DD / CVaR / stability / costs / capacity / regime fails
        └── approve only if all configured checks pass
                │
                ▼
PaperTradingInterface.from_result → PaperTradingConfig
   (experiment_id, seed, config, lineage, scorecard, gates)
```

### Validation gates

#### `GateThresholds`

| Field | Default | Role |
|-------|---------|------|
| `require_out_of_sample` | `True` | OOS metric mandatory |
| `min_oos_sharpe` | `0.0` | Floor on OOS Sharpe when present |
| `min_sharpe` | `None` | Optional overall Sharpe floor |
| `max_drawdown` | `0.35` | Drawdown ceiling |
| `max_cvar` | `None` | Tail risk ceiling |
| `min_stability` | `None` | Rolling stability floor |
| `min_regime_robustness` | `None` | Cross-regime floor |
| `max_turnover` / `max_transaction_costs` | `None` | Implementation burden |
| `min_capacity` | `None` | Capital capacity floor |
| `reject_in_sample_only` | `True` | High IS Sharpe without OOS → reject |
| `min_statistical_confidence` | `None` | Optional statistical evidence |

#### `GateResult`

- `approved: bool`
- `out_of_sample_ok: bool`
- `checks: dict[str, bool]`
- `reasons: list[str]`
- `scorecard: dict`
- `policy` string documenting the never-Sharpe-alone rule

#### `evaluate_gates`

```python
from iqrp.app.backtesting import evaluate_gates, GateThresholds
from iqrp.app.backtesting.performance import StrategyScorecard

# Policy check baked into Phase 13 validator:
sc = StrategyScorecard(sharpe=3.0, total_return=1.0, out_of_sample=None)
gate = evaluate_gates(sc, in_sample_sharpe=3.0)
assert gate.approved is False
assert gate.out_of_sample_ok is False
```

```python
from iqrp.app.backtesting import BacktestEngine

eng = BacktestEngine()
result = eng.run(returns=rets, signals=sigs, oos_fraction=0.25)
gate = eng.validate_for_promotion(
    result,
    gates=GateThresholds(min_oos_sharpe=0.0, max_drawdown=0.35),
)
print(gate.approved, gate.reasons)
```

Invalidated experiments always fail:

```python
if result.invalidated:
    gate = eng.validate_for_promotion(result)
    # approved=False, reasons include invalidation
```

Helpers: `require_oos(scorecard)`, `summarize_gate_policy()`.

### Scorecard (promotion input)

`StrategyScorecard` / `build_scorecard` supply the gate inputs. See [PerformanceMetrics](PerformanceMetrics.md).

Critical fields for gates:

- `out_of_sample` — **required** when `require_out_of_sample=True`
- `sharpe`, `max_drawdown`, `cvar`, `stability`, `regime_robustness`
- `turnover`, `transaction_costs`, `capacity`

`StrategyScorecard.passes_gates(...)` is a convenience check; production promotion should use `evaluate_gates` for the OOS / in-sample-only policy guards.

### Paper trading handoff

```python
from iqrp.app.backtesting import PaperTradingInterface, PaperTradingConfig

# Preferred: through the engine (runs gates, then packages)
pt: PaperTradingConfig = eng.to_paper_trading(result)

# Or from registry id
pt = eng.to_paper_trading(result.experiment_id)

print(pt.experiment_id, pt.seed, pt.lineage, pt.gates)
```

`PaperTradingConfig` fields:

| Field | Purpose |
|-------|---------|
| `experiment_id` | Traceability to registry |
| `strategy_name` | Human label |
| `seed` | RNG continuity |
| `config` | Strategy / backtest config snapshot |
| `lineage` | Data/feature/model/risk/portfolio/execution/code versions |
| `scorecard` | Metrics at promotion time |
| `gates` | GateResult dict |
| `created_at` | UTC timestamp |
| `notes` | Reminder to preserve versions live |

`PaperTradingInterface` tracks configs in-memory (`from_result`, `from_experiment`, `get`, `list`).

**Rule:** paper trading must preserve strategy version, feature/model versions, and execution config from the promoting backtest so live paper remains comparable to the validated experiment.

### Critical rules (Phase 13 validation)

| # | Rule |
|---|------|
| 1 | **Out-of-sample evidence is mandatory** for promotion (`require_out_of_sample=True`). |
| 2 | **Never promote on highest historical return or Sharpe alone.** |
| 3 | High in-sample Sharpe with missing OOS is explicitly rejected (`reject_in_sample_only`). |
| 4 | Invalidated / leakage-contaminated experiments cannot be promoted. |
| 5 | Gates cover risk, drawdown, capacity, costs, stability, regime, and optional statistics — not a single ratio. |
| 6 | Paper handoff preserves lineage versions and seed. |
| 7 | Rejected experiments remain in the experiment registry for audit. |

### Integration

- Walk-forward / rolling retrain / `oos_fraction` generate the OOS series that populates `scorecard.out_of_sample`
- Capacity testing can populate `scorecard.capacity` and `min_capacity` gates
- Scenario regime metrics can populate `regime_robustness`
- Execution / Risk / Portfolio participate only via imported versions in lineage — Phase 13 does not modify those packages

### Example: end-to-end promote-or-reject

```python
import numpy as np
from iqrp.app.backtesting import BacktestEngine, GateThresholds

rng = np.random.default_rng(0)
rets = rng.normal(0.0005, 0.01, 504)
sigs = np.tanh(np.cumsum(rets) / 10)

eng = BacktestEngine()
result = eng.run(returns=rets, signals=sigs, seed=0, oos_fraction=0.2, name="demo")
gate = eng.validate_for_promotion(
    result,
    GateThresholds(min_oos_sharpe=0.0, max_drawdown=0.5),
)
if gate.approved:
    paper = eng.to_paper_trading(result)
    eng.save("artifacts/promoted.json", result)
else:
    print("rejected:", gate.reasons)
```
