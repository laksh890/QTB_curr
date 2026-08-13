# Reproducibility

Experiment registry with full lineage (data/feature/model/risk/portfolio/execution/code versions + seed). PIT leakage detection. Serializer for audit persistence.

---

## Purpose

Every institutional backtest must be reconstructible and auditable. Reproducibility covers lineage fingerprints, experiment retention (including rejects), point-in-time leakage detection, and JSON serialization.

**Modules:** `experiment_registry`, `pit`, `serializer`, `config.ReproducibilityConfig`  
**Related:** [BacktestingPlatform](BacktestingPlatform.md) · [EventEngine](EventEngine.md) · [StrategyValidation](StrategyValidation.md) · [RollingRetraining](RollingRetraining.md)

---

## Architecture

```text
BacktestSettings.reproducibility
        │
ExperimentLineage (versions + seed)
        │
ExperimentRegistry ── create / update / invalidate / list / persist
        │
BacktestResult + serialize_result / save_json
        │
PIT helpers: assert_no_lookahead, detect_leakage, filter_universe_asof
```

---

## Key APIs

### Lineage

```python
from iqrp.app.backtesting.experiment_registry import ExperimentLineage
from iqrp.app.backtesting.config import BacktestSettings

settings = BacktestSettings.default()
lineage = ExperimentLineage.from_settings(settings, seed=7)
# data_version, feature_version, label_version, model_version,
# risk_version, portfolio_version, execution_version, code_version, seed
```

Configured via Hydra / `ReproducibilityConfig`:

```yaml
reproducibility:
  seed: 42
  data_version: "1.0.0"
  feature_version: "1.0.0"
  label_version: "1.0.0"
  model_version: "1.0.0"
  risk_version: "1.0.0"
  portfolio_version: "1.0.0"
  execution_version: "1.0.0"
  code_version: "1.0.0"
```

### `ExperimentRegistry`

```python
from iqrp.app.backtesting import ExperimentRegistry
from iqrp.app.backtesting.types import BacktestState

reg = ExperimentRegistry()
rec = reg.create(name="mom_v1", lineage=lineage, config={"lookback": 20})
reg.update_state(rec.experiment_id, BacktestState.RUNNING)
reg.register_result(rec.experiment_id, state=BacktestState.COMPLETED, metrics={"sharpe": 0.8})
reg.invalidate(rec.experiment_id, "leakage detected")  # retained for audit
reg.list(include_invalidated=True)
```

`ExperimentRecord` stores id, name, state, timestamps, lineage, config, metrics, warnings, invalidation reason, tags, result summary. Rejected / invalidated experiments are **retained**.

### PIT / leakage

```python
from iqrp.app.backtesting.pit import (
    assert_no_lookahead,
    detect_leakage,
    filter_universe_asof,
    filter_frame_asof,
    available_asof,
    LookaheadViolation,
)

assert_no_lookahead(data_ts, event_ts, context="close")
universe = filter_universe_asof({"AAA": (start, end)}, asof)
report = detect_leakage([0, 1, 2], [0, 1, 3], timestamps=[1, 2, 3])
if report.has_leakage:
    # BacktestEngine → INVALIDATED
    ...
```

Survivorship guard: `filter_universe_asof` excludes not-yet-listed and already-delisted symbols at `asof`.

### Serializer

```python
from iqrp.app.backtesting.serializer import (
    serialize_result,
    deserialize_result,
    save_json,
    load_json,
    to_jsonable,
)

payload = serialize_result(result)
save_json("artifacts/run.json", payload)
restored = deserialize_result(load_json("artifacts/run.json"))
```

`BacktestEngine.save` / `load` persist result + registry records together.

### Deterministic event path

Event queue ordering `(timestamp, priority, sequence)` plus `BacktestClock` and seeded scenario/MC paths make event-driven runs bit-stable given the same inputs.

---

## Critical rules

| Rule | Detail |
|------|--------|
| Record versions + seed every run | Incomplete lineage is a process failure |
| Retain rejects | Invalidated/failed experiments stay in the registry |
| Naive timestamps forbidden | PIT checks and events require tz-aware datetimes |
| Leakage → INVALIDATED | Do not “fix” and continue |
| Same seed, same path | Stochastic extensions (MC) must pass seed explicitly |
| Paper inherits lineage | PaperTradingConfig copies versions from the promoting experiment |

---

## Integration

- Lineage versions should mirror Feature Store / Model Lifecycle / Execution config versions when those systems are used (by value, via imports/settings — no cross-package mutation)
- Corporate actions use `actions_asof` for PIT filtering
- Phase 13 validator checks gate policy and docs presence

---

## Example: save / reload audit package

```python
from iqrp.app.backtesting import BacktestEngine

eng = BacktestEngine()
result = eng.run(returns=rets, signals=sigs, seed=42, name="audit_demo")
path = eng.save("/tmp/bt_audit.json", result)
loaded = eng.load(path)
assert loaded.experiment_id == result.experiment_id
assert loaded.lineage.seed == 42
```

---

## Operational runner reproducibility (Phase 13 executable layer)

The operational `BacktestRunner` path records a concrete reproducibility bundle distinct from (but compatible with) research `ExperimentLineage`:

| Artifact | Where |
|----------|--------|
| Full run config | `results/{backtest_id}/configuration/config.json` and `OperationalBacktestResult.config` |
| Dataset path / id / version | `BacktestRunConfig.dataset_*`; report `reproducibility.dataset` |
| Dataset checksum | `DatasetRegistry` / `DatasetRecord.checksum` (`file_sha256` or `parquet_canonical_sha256`) |
| Seed | `BacktestRunConfig.seed` → execution sim + result/`report` reproducibility block |
| Code / backend versions | Portfolio & execution adapter `backend` names in diagnostics; platform `code_version` via research settings when used together |
| Lifecycle audit | `diagnostics.lifecycle` history |
| Capital reconciliation | `diagnostics/reconciliation.json` |

### Reconstruct a run

1. Pin the same local dataset bytes (verify `DatasetRegistry.verify_checksum`).
2. Load the same YAML/dict config (`BacktestRunConfig.from_yaml` / persisted `config.json`).
3. Register the same `strategy_id` / `strategy_version` (and identical `strategy_params`).
4. Use the same `seed`.
5. Use the same package/code revision (git SHA or release `code_version`).
6. Re-run: `BacktestRunner(cfg).validate(); prepare(); run()`.

Expect the same equity path and order/fill counts when adapters resolve to the same backends and data checksum matches. Parallel sweeps assign per-experiment seeds (`seed0 + i`) and unique `backtest_id`s so workers do not share mutable state.

### Report block

`build_report_payload` always includes:

```json
{
  "reproducibility": {
    "seed": 42,
    "strategy": {"id": "buy_and_hold", "version": "1.0.0"},
    "dataset": {"path": "...", "id": "...", "version": "..."},
    "backends": {"portfolio": "...", "execution": "..."}
  }
}
```

See [BacktestRunner](BacktestRunner.md) · [ResultStorage](ResultStorage.md) · [UserGuide](UserGuide.md) · [DatasetValidation](DatasetValidation.md).
