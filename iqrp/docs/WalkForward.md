# Walk-Forward

Causal rolling / expanding / anchored folds with purge and embargo. Training never sees future data.

---

## Purpose

Walk-forward evaluation produces **out-of-sample** fold metrics under strict causality. It is the primary mechanism for generating OOS evidence required by promotion gates.

**Package:** `iqrp.app.backtesting.walk_forward`  
**Primary type:** `WalkForwardEngine`  
**Related:** [BacktestingPlatform](BacktestingPlatform.md) · [RollingRetraining](RollingRetraining.md) · [StrategyValidation](StrategyValidation.md) · [Reproducibility](Reproducibility.md)

---

## Architecture

```text
generate_windows(mode, purge, embargo)
        │
        ▼
WalkForwardWindow (train / optional val / test indices)
        │
 assert_no_future_training
        │
        ▼
WalkForwardEvaluator.evaluate(fold_fn)
        │
        ▼
WalkForwardReport (per-fold + aggregated OOS metrics)
```

Modes (`WindowMode`):

| Mode | Train behavior |
|------|----------------|
| `rolling` | Fixed-length train window slides forward |
| `expanding` | Train grows from index 0 |
| `anchored` | Train grows from fixed `anchor` |
| `purged_kfold` | Contiguous purged K-fold (+ embargo) |

---

## Key APIs

### Window types

- `TrainingWindow(start, end)` — half-open `[start, end)`
- `ValidationWindow` — carved from end of train, still before test
- `TestWindow` — OOS fold; `prediction_timestamp` is test start
- `WalkForwardWindow` — fold_id, mode, train/test/validation, purge/embargo, index arrays

### `generate_windows`

```python
from iqrp.app.backtesting.walk_forward import generate_windows

wins = generate_windows(
    n=500,
    train_size=252,
    test_size=21,
    mode="rolling",
    step=21,
    purge=5,
    embargo=5,
    validation_size=0,
)
```

Parameters:

- `purge` — bars excluded between train end and test start (label-horizon gap); also purges train indices near the test fold
- `embargo` — bars after test end excluded from training (serial leakage guard)
- `n_splits` — used for `purged_kfold`

### Purge and embargo

```python
from iqrp.app.backtesting.walk_forward import (
    apply_purge,
    apply_embargo,
    purged_kfold_splits,
    embargo_splits,
)

# Remove train indices whose label horizon overlaps the test fold
purged = apply_purge(train_idx, test_start=100, test_end=120, purge=5)
final = apply_embargo(purged, test_idx, embargo=5, purge=5)

splits = purged_kfold_splits(n=500, n_splits=5, purge=5)
```

Lopez de Prado–style purged CV: observations whose forward-looking label overlaps the test fold leave the training set. Embargo blocks autocorrelation bleed after the test fold.

### `WalkForwardEngine`

```python
from iqrp.app.backtesting.walk_forward import WalkForwardEngine
import numpy as np

eng = WalkForwardEngine()

def evaluate_fold(train_idx, test_idx):
    # Fit only on train_idx; score only on test_idx
    return {"n_train": len(train_idx), "n_test": len(test_idx)}

report = eng.run(
    n=200,
    train_size=80,
    test_size=20,
    step=20,
    mode="expanding",
    purge=2,
    embargo=2,
    evaluate_fold=evaluate_fold,
)
# report is a dict (as_dict=True default) with folds + aggregates
```

Also:

- `windows(...)` — generate + assert causality
- `run_on_windows(windows, evaluate_fold)`
- `run_arrays(X, y, fit_predict=...)` — convenience over array rows

### Evaluator / report

- `FoldResult`, `WalkForwardReport`, `aggregate_fold_metrics`
- `assert_no_future_training(windows)` raises if `max(train) >= min(test)` for causal modes

### Via `BacktestEngine`

```python
from iqrp.app.backtesting import BacktestEngine

bt = BacktestEngine()
wf = bt.walk_forward(returns=rets, train_size=100, test_size=20, mode="rolling")
```

Uses settings from `BacktestSettings.walk_forward` when sizes are omitted.

---

## Critical rules

| Rule | Detail |
|------|--------|
| No future training | For causal modes, `train.end <= test.start` after purge; `max(train_idx) < prediction_timestamp` |
| OOS is the product | Fold metrics are evaluated on test indices only |
| Purge label horizons | Overlapping forward labels must leave the train set |
| Embargo serial dependence | Post-test bars excluded from alternate/next train samples |
| Validation stays causal | Inner validation is carved from train and remains before test |
| purged_kfold note | May retain non-overlapping post-test train samples by design; causal modes forbid this |

---

## Integration

- Feeds OOS Sharpe into `StrategyScorecard.out_of_sample` / promotion gates
- Complements [RollingRetraining](RollingRetraining.md) (episode-level OOS) without replacing fold-based WF
- Does not modify Portfolio / Risk / Execution packages; fold callbacks may import their estimators

---

## Example: purged rolling Sharpe

```python
import numpy as np
from iqrp.app.backtesting.walk_forward import WalkForwardEngine
from iqrp.app.backtesting.performance import sharpe_ratio

rng = np.random.default_rng(7)
rets = rng.normal(0.0005, 0.01, 400)

def fold(tr, te):
    return {"sharpe": sharpe_ratio(rets[te]), "n_test": int(len(te))}

report = WalkForwardEngine().run(
    n=len(rets),
    train_size=120,
    test_size=20,
    purge=5,
    embargo=5,
    mode="rolling",
    evaluate_fold=fold,
)
print(report["aggregate"])  # aggregated OOS metrics
```
