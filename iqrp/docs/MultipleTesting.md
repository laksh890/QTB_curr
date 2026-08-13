# Multiple Testing

Family-wise and false-discovery control for alpha research, with explicit trial tracking so silent search inflation cannot masquerade as discovery.

**Package:** `iqrp.app.alpha.statistical_validation.multiple_testing`  
**Related:** [SignalValidation](SignalValidation.md) · [False discovery helpers](StatisticalValidation.md) · [AlphaResearch](AlphaResearch.md)

---

## Why it matters

Alpha research is a multiple-comparison problem: hundreds of lookbacks, features, formulas, and regimes are tried. A raw p < 0.05 on the “best” candidate after untracked search is usually a **false discovery**.

Rules restated:

- Statistical significance alone ≠ alpha
- Historical Sharpe alone cannot approve
- Trial budget must enter DSR (`n_trials`) and p-value adjustment

---

## Methods

`multiple_testing_adjustment` supports:

| `method` | Control | Behavior |
|----------|---------|----------|
| `bonferroni` | FWER | `p_adj = min(p * m, 1)` |
| `holm` | FWER (step-down) | More powerful than Bonferroni; still family-wise |
| `fdr_bh` | FDR (Benjamini–Hochberg) | Default in engine `validate` |
| `none` | — | Pass-through (debug only; not for promotion) |

```python
import numpy as np
from iqrp.app.alpha.statistical_validation.multiple_testing import (
    multiple_testing_adjustment,
    get_experiment_tracker,
)

pvalues = [0.001, 0.02, 0.04, 0.20, 0.50]
out = multiple_testing_adjustment(
    pvalues,
    method="fdr_bh",
    alpha=0.05,
    label="momentum_screen_v3",
)
# out["adjusted"], out["rejected"], out["n_experiments"], out["raw_pvalues"]

holm = multiple_testing_adjustment(pvalues, method="holm", label="holm_check")
bonf = multiple_testing_adjustment(pvalues, method="bonferroni", label="bonf_check")
```

When `iqrp.app.timeseries.multiple_testing.adjust_pvalues` is importable it is preferred; otherwise an equivalent local implementation is used.

---

## Experiment / trial tracking

`ExperimentTracker` accumulates the number of hypothesis tests in a research session so later DSR and reporting reflect search intensity.

```python
from iqrp.app.alpha.statistical_validation.multiple_testing import (
    ExperimentTracker,
    get_experiment_tracker,
)

tracker = get_experiment_tracker()  # process-global default
tracker.reset()  # start of a clean research session

multiple_testing_adjustment(
    [0.01, 0.03],
    method="fdr_bh",
    tracker=tracker,
    label="cs_value_screen",
    record=True,
)
assert tracker.n_experiments >= 2
print(tracker.history[-1])
# {"label": "cs_value_screen", "n_added": 2, "n_experiments": ..., "meta": {...}}
```

`AlphaResearchEngine.validate` records trials via this tracker and pads the p-value vector to the declared `n_trials` budget so selection intensity is not understated.

```python
from iqrp.app.alpha import AlphaResearchEngine

eng = AlphaResearchEngine()
val = eng.validate(signal, returns, n_trials=40, mt_method="fdr_bh", label="eng_validate")
mt = val["multiple_testing"]
assert mt["n_tests_this_call"] == 40 or mt["n_experiments"] >= 1
```

Use a private `ExperimentTracker()` per research project when you need isolation from the global counter.

---

## False discovery practice

1. **Declare the family** before peeking (feature list, lookback grid, formula library).
2. **Record every test** (`record=True`) — including discarded screens.
3. **Adjust** with BH-FDR for discovery screens; prefer Holm/Bonferroni for small confirmatory families.
4. **Feed `n_trials` into DSR** and keep rejected experiments in `SignalRegistry` so the institution remembers what was tried.
5. **Do not promote** on raw p-values after large untracked search.

```python
from iqrp.app.alpha.statistical_validation import storey_qvalues, false_discovery_report
from iqrp.app.alpha.statistical_validation.deflated_sharpe import deflated_sharpe_ratio

q = storey_qvalues(pvalues, lambda_=0.5)
fdr = false_discovery_report(pvalues, method="fdr_bh", alpha=0.05)
# fdr["n_discoveries"], fdr["qvalues"], fdr["pi0"], fdr["n_experiments"]

# Deflate observed Sharpe with the tracked trial count
tracker = get_experiment_tracker()
dsr = deflated_sharpe_ratio(
    obs_sr=1.2,
    n_trials=max(tracker.n_experiments, 1),
    n_obs=500,
    return_details=True,
)
```

Preserve rejects:

```python
from iqrp.app.alpha import get_default_registry, SignalStatus

reg = get_default_registry()
# After rejecting a lead that failed BH-FDR:
# eng.registry.transition(eid, SignalStatus.REJECTED, reason="BH-FDR not rejected", actor="research")
assert reg.list_experiments(status=SignalStatus.REJECTED, include_rejected=True) is not None
```

---

## Choosing a procedure

| Context | Prefer |
|---------|--------|
| Broad feature / formula screen | `fdr_bh` |
| Small confirmatory set pre-registered | `holm` or `bonferroni` |
| Single pre-registered hypothesis | classical test; still document `n_trials=1` |
| Engine default in `validate` | `fdr_bh` |

Adjusted significance is still not alpha. Continue to [SignalValidation](SignalValidation.md) approve gates, [BacktestValidation](BacktestValidation.md), and [SignalLifecycle](SignalLifecycle.md).
