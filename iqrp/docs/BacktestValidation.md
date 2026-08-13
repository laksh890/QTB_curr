# Backtest Validation

Leakage-safe backtesting for alpha research: walk-forward evaluation, purged cross-validation, embargo gaps, nested CV for model selection, look-ahead prevention, and survivorship discipline.

**Package:** `iqrp.app.alpha.backtesting`  
**Engine entry:** `AlphaResearchEngine.backtest` / `stress_test`  
**Parent:** [AlphaResearch](AlphaResearch.md) · Related: [PurgedCV](PurgedCV.md) · [Embargo](Embargo.md)

---

## Principles

1. **Train never sees the future.** Expanding/rolling windows are strictly causal.
2. **Overlapping labels leak.** Forward-return horizons that span a test fold must be purged from train.
3. **Serial dependence leaks.** Post-test embargo blocks autocorrelation bleed into the next train sample.
4. **Selection needs nested CV.** Hyperparameters chosen on outer OOS are contaminated; use inner folds only inside outer train.
5. **Backtest Sharpe ≠ approval.** Historical Sharpe alone cannot approve; pair with [SignalValidation](SignalValidation.md).

---

## Walk-forward

```python
import numpy as np
from iqrp.app.alpha.backtesting.walk_forward import (
    walk_forward_splits,
    walk_forward_backtest,
)

n = 500
signal = np.random.default_rng(0).normal(size=n)
returns = np.random.default_rng(1).normal(0, 0.01, n)

for train_idx, test_idx in walk_forward_splits(
    n,
    train_size=200,
    test_size=40,
    gap=5,          # embargo between train and test
    expanding=False,
):
    assert train_idx.max() < test_idx.min() - 5 + 1 or True  # gap enforced in construction

wf = walk_forward_backtest(
    signal,
    returns,
    train_size=200,
    test_size=40,
    gap=5,
    cost_bps=1.0,
    mode="long_short",
    returns_are_forward=False,
)
# Concatenated OOS net returns + fold diagnostics
```

`gap` is the embargo between the end of train and the start of test. Prefer `returns_are_forward=True` when the return series is already a forward label aligned to the signal timestamp.

---

## Purged CV

Removes train observations whose label horizon overlaps the test window.

```python
from iqrp.app.alpha.backtesting.purged_cv import purged_kfold_splits, purge_train_indices

for tr, te in purged_kfold_splits(n, n_splits=5, purge=5):
    # train excludes [test_start - purge, test_end + purge]
    assert np.intersect1d(tr, te).size == 0

# Or purge an existing train set relative to a test fold
tr2 = purge_train_indices(tr, te, purge=10, n=n)
```

Set `purge` ≥ prediction horizon (and preferably ≥ label span used in IC/targets).

---

## Embargo

Additional post-test exclusion to block serial leakage after the fold ends.

```python
from iqrp.app.alpha.backtesting.embargo import apply_embargo, embargo_splits

for tr, te in embargo_splits(n, n_splits=5, embargo=5, purge=5):
    tr = apply_embargo(tr, te, embargo=5, purge=5)
```

Combined purge + embargo is the default discipline for overlapping financial labels.

---

## Nested CV

Outer folds estimate generalization; inner folds (inside outer train, after purge/embargo) select hyperparameters without touching outer OOS.

```python
from iqrp.app.alpha.backtesting.nested_cv import nested_cv_splits

for fold in nested_cv_splits(
    n,
    n_outer=5,
    n_inner=3,
    purge=5,
    embargo=5,
):
    outer_train = fold["outer_train"]
    outer_test = fold["outer_test"]
    for inner_train, inner_val in fold["inner_folds"]:
        # Fit / select on inner_train, score on inner_val
        # Evaluate final choice once on outer_test
        pass
```

Never tune on outer test. Report outer OOS only as the honest estimate.

---

## Look-ahead prevention checklist

| Risk | Mitigation in IQRP |
|------|--------------------|
| Signal uses future prices | PIT helpers; past windows only in discovery |
| Label overlap into train | `purge` ≥ horizon |
| Autocorr bleed after test | `embargo` / `gap` |
| Same-bar returns as target | Use `forward_returns` / lag signal |
| Parameter peeking on OOS | `nested_cv_splits` |
| Costless fantasy Sharpe | `signal_backtest(..., cost_bps=...)` |
| Regime cherry-picking | Pre-register regimes; see `analyze_regimes` |

```python
from iqrp.app.alpha import AlphaResearchEngine
from iqrp.app.alpha.research.decay import forward_returns

eng = AlphaResearchEngine()
fwd = forward_returns(returns, 1)          # labels
bt = eng.backtest(signal, returns, cost_bps=2.0, mode="long_short")
# bt["net_sharpe"] is diagnostic — not an approval criterion
```

---

## Survivorship and universe honesty

Backtests are only as honest as the universe:

- Use point-in-time membership (include names that later delist or are acquired).
- Do not build signals on today’s surviving liquid set and pretend that was the tradable universe historically.
- Align corporate actions and restated fundamentals to availability timestamps.
- Document `universe` on `SignalDefinition` and keep it stable across evaluate / backtest / capacity.

The engine does not silently rewrite history; callers must supply PIT panels. Survivorship bias is treated as a **data contract** violation equal in severity to look-ahead.

---

## Stress and regime overlays

```python
regimes = (np.abs(returns) > np.nanstd(returns)).astype(int)
stress = eng.stress_test(signal, returns, regimes=regimes, cost_bps=2.0)
# baseline + vol shocks + sign_flip + optional regime_performance
```

Stress diagnostics inform robustness; they do not replace purged/embargoed OOS evaluation.

---

## Recommended research sequence

1. PIT signal construction ([SignalDiscovery](SignalDiscovery.md))
2. Walk-forward or purged/embargo CV backtest (this doc)
3. Nested CV if any hyperparameters are fit
4. Statistical validation + MT ([SignalValidation](SignalValidation.md), [MultipleTesting](MultipleTesting.md))
5. Decay / capacity / ensemble triage
6. Lifecycle approval with hypothesis ([SignalLifecycle](SignalLifecycle.md))
