# Rolling Retraining

Schedule-driven retrains with versioned feature/model/parameter snapshots and OOS episode evaluation.

---

## Purpose

Rolling retraining simulates production model refresh: triggers fire on a schedule (time, performance, drift, regime, or composite), models are fit on **past-only** windows, registered with versions, and scored on subsequent OOS bars.

**Package:** `iqrp.app.backtesting.rolling_retraining`  
**Primary type:** `RollingRetrainer`  
**Related:** [WalkForward](WalkForward.md) · [Reproducibility](Reproducibility.md) · [BacktestingPlatform](BacktestingPlatform.md)

---

## Architecture

```text
RetrainSchedule (Time / Performance / Drift / Regime / Composite)
        │
        ▼
RollingRetrainer.maybe_retrain(t)
   train on [start, t) only
   FeatureSnapshot + ParameterSnapshot
   ModelRegistry.register(trained_through = t-1)
        │
        ▼
Predict / score bars with trained_through < t
        │
        ▼
RetrainEpisode segments → RollingRetrainReport
```

**Invariant:** a model with `trained_through=t` may only be evaluated on indices `> t`. Retrains never include the prediction bar.

---

## Key APIs

### Triggers and schedule

```python
from iqrp.app.backtesting.rolling_retraining import (
    RetrainSchedule,
    TimeTrigger,
    PerformanceTrigger,
    DriftTrigger,
    RegimeTrigger,
    CompositeTrigger,
)

schedule = RetrainSchedule(every=20)  # default TimeTrigger
# Or compose:
schedule = RetrainSchedule(
    trigger=CompositeTrigger([
        TimeTrigger(every=63),
        PerformanceTrigger(metric="sharpe", min_value=0.0, lookback=20),
    ])
)
```

| Trigger | Fires when |
|---------|------------|
| `TimeTrigger` | Bars since last retrain ≥ `every` |
| `PerformanceTrigger` | Rolling metric below `min_value` |
| `DriftTrigger` | Feature/prediction drift exceeds threshold (context-driven) |
| `RegimeTrigger` | Regime label change / transition in context |
| `CompositeTrigger` | Any/all child triggers (configurable) |

`TriggerDecision` carries `should_retrain`, `kind`, `reason`, `details`.

### Snapshots and model registry

- `FeatureSnapshot` / `FeatureSnapshotStore` — versioned feature matrices for the train slice
- `ParameterSnapshot` / `ParameterSnapshotStore` — hyperparameter fingerprints
- `ModelRegistry` / `ModelSnapshot` — version, `trained_through`, feature/parameter versions, trigger, metrics, activate/get/active

### `RollingRetrainer`

```python
from iqrp.app.backtesting.rolling_retraining import RollingRetrainer, RetrainSchedule
import numpy as np

X = np.random.randn(300, 4)
y = X[:, 0] + 0.1 * np.random.randn(300)

def train_fn(X_tr, y_tr, params):
    return {"beta": float(np.linalg.lstsq(X_tr, y_tr, rcond=None)[0][0])}

def predict_fn(model, X_te):
    return X_te[:, 0] * model["beta"]

def score_fn(model, X_te, y_te):
    pred = predict_fn(model, X_te)
    return {"mse": float(np.mean((pred - y_te) ** 2))}

retrainer = RollingRetrainer(
    schedule=RetrainSchedule(every=40),
    train_window=100,  # None → expanding from origin
    origin=0,
)
report = retrainer.run(
    X=X, y=y, train_fn=train_fn, predict_fn=predict_fn, score_fn=score_fn
)
```

Key methods:

| Method | Role |
|--------|------|
| `training_slice(t)` | Half-open `[start, t)` — end exclusive |
| `maybe_retrain(...)` | Evaluate schedule; train/register if fired |
| `predict_at(t, ...)` | Predict with model trained strictly before `t` |
| `run(...)` | Walk time, score OOS, close episodes on retrain |
| `active_model()` | Current active model object |

`RetrainEvent` records `(t, decision, snapshot)`. `RetrainEpisode` spans `(eval_start, eval_end)` per model version.

### Evaluator

- `RollingRetrainEvaluator`, `RollingRetrainReport`, `aggregate_episode_metrics`

### Via `BacktestEngine`

```python
from iqrp.app.backtesting import BacktestEngine

bt = BacktestEngine()
out = bt.retrain_rolling(X=X, y=y, every=20, train_window=80)
```

---

## Critical rules

| Rule | Detail |
|------|--------|
| No future training | Train bounds end at `t` exclusive; `trained_through = end - 1` must be `< t` |
| No future prediction | `predict_at` / scoring require `trained_through < t` |
| Version everything | Feature, parameter, and model snapshots are linked on register |
| OOS episodes | Metrics accumulate only on bars after the model’s train cutoff |
| Triggers are causal | Context at `t` must not leak future performance labels into the decision beyond declared inputs |

---

## Integration

- Model versions feed `ExperimentLineage.model_version` / paper-trading handoff
- Import estimators from forecasting / alpha packages inside `train_fn`; do not edit those packages from backtesting
- Complements walk-forward: WF = fixed folds; rolling retrain = production refresh simulation

---

## Example: force warm-start then timed refreshes

```python
retrainer = RollingRetrainer(schedule=RetrainSchedule(every=25), train_window=50)
# warm_start_train=True (default) forces first model at t0
report = retrainer.run(X=X, y=y, train_fn=train_fn, score_fn=score_fn)
assert report["n_models"] >= 1
assert report["active_version"] is not None
```
