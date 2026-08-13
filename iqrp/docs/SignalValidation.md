# Signal Validation

Statistical and research validation for alpha candidates: Information Coefficient (IC), Rank IC, classical significance, bootstrap confidence intervals, permutation tests, Deflated Sharpe Ratio (DSR), Probability of Backtest Overfitting (PBO), and stability diagnostics.

**Package:** `iqrp.app.alpha.statistical_validation` · `iqrp.app.alpha.research`  
**Engine entry:** `AlphaResearchEngine.evaluate` / `AlphaResearchEngine.validate`  
**Parent:** [AlphaResearch](AlphaResearch.md) · Related: [MultipleTesting](MultipleTesting.md) · [InformationCoefficient](InformationCoefficient.md)

---

## Governance

Validation **informs** research; it does not approve alpha by itself.

- Statistical significance alone ≠ alpha
- Historical Sharpe alone cannot approve
- `validate` evidence is required before `approve()` when using the engine gates

Alignment rule: `signal[t]` must be known strictly before `forward_returns[t]` realizes. Pass forward labels (or set `returns_are_forward=True`), never contemporaneous unmarked returns as features.

---

## Metrics

| Metric | Module / function | Meaning |
|--------|-------------------|---------|
| IC | `compute_ic` | Pearson correlation of signal vs forward returns |
| Rank IC | `compute_rank_ic` | Spearman / rank correlation (outlier-robust) |
| Significance | `ic_significance` | t-test (and HAC variants) vs H0: IC = 0 |
| Bootstrap CI | `iid_bootstrap_ci` | Resampled CI for IC (or other stats) |
| Permutation | `permutation_ic_test` | Null distribution by shuffling labels |
| DSR | `deflated_sharpe_ratio` | Sharpe deflated for selection bias / non-normality |
| PBO | `probability_backtest_overfitting` | Probability that best in-sample strategy underperforms OOS |
| Stability | evaluator stability windows | Rolling consistency of predictive power |
| Multiple testing | `multiple_testing_adjustment` | Bonferroni / Holm / BH-FDR — see [MultipleTesting](MultipleTesting.md) |

---

## Engine workflow

```python
import numpy as np
from iqrp.app.alpha import AlphaResearchEngine, SignalDefinition

rng = np.random.default_rng(7)
returns = rng.normal(0, 0.01, 600)
signal = np.concatenate([[0.0], returns[:-1]])  # lag-1 PIT signal

eng = AlphaResearchEngine()
defn = SignalDefinition(
    name="lag_return",
    version="0.1.0",
    formula="returns.shift(1)",
    features=("returns",),
    lookback=1,
    horizon=1,
    universe="default",
    frequency="1d",
    direction="long_short",
    expected_relationship="positive",
    economic_hypothesis=(
        "One-bar autocorrelation from microstructure and slow inventory "
        "adjustment in liquid names; expect fast IC decay."
    ),
    owner="research",
    signal_type="momentum",
)
rec = eng.register(defn, signal=signal)

# Predictive evaluation (IC family + stability → research report)
eval_out = eng.evaluate(
    signal,
    returns,
    experiment_id=rec.experiment_id,
    definition=defn,
)
print(eval_out["ic_mean"])

# Full statistical validation bundle
val = eng.validate(
    signal,
    returns,
    experiment_id=rec.experiment_id,
    n_trials=30,
    mt_method="fdr_bh",
    seed=7,
)
assert "significance" in val
assert "bootstrap" in val
assert "permutation" in val
assert "multiple_testing" in val
assert "deflated_sharpe" in val
assert "pbo" in val
```

`validate` attaches `diagnostics["validation"]` and `diagnostics["validate"]=True` so `approve()` can detect evidence.

---

## IC and Rank IC

```python
from iqrp.app.alpha.research.information_coefficient import compute_ic
from iqrp.app.alpha.research.rank_ic import compute_rank_ic
from iqrp.app.alpha.research.decay import forward_returns

fwd = forward_returns(returns, horizon=1)
ic = compute_ic(signal, fwd)
rank_ic = compute_rank_ic(signal, fwd)
```

Prefer Rank IC for fat-tailed cross-sections; use both when comparing time-series vs CS signals.

---

## Significance, bootstrap, permutation

```python
from iqrp.app.alpha.statistical_validation.significance import ic_significance
from iqrp.app.alpha.statistical_validation.bootstrap import iid_bootstrap_ci
from iqrp.app.alpha.statistical_validation.permutation import permutation_ic_test

sig = ic_significance(signal, fwd)
# {"ic", "t_stat", "pvalue", "n", "method", ...}

boot = iid_bootstrap_ci(signal, fwd, stat="ic", n_boot=200, seed=0)
# confidence interval under resampling

perm = permutation_ic_test(signal, fwd, n_perm=200, seed=0)
# empirical p-value under label shuffle
```

Interpretation discipline:

- Low classical p-value after many untracked trials is **not** discovery.
- Always feed trial count into multiple-testing adjustment and DSR (`n_trials`).

---

## Deflated Sharpe (DSR) and PBO

```python
from iqrp.app.alpha.statistical_validation.deflated_sharpe import deflated_sharpe_ratio
from iqrp.app.alpha.statistical_validation.probability_backtest_overfitting import (
    probability_backtest_overfitting,
)
from iqrp.app.alpha.backtesting.signal_backtest import signal_backtest

bt = signal_backtest(signal, returns, cost_bps=1.0, returns_are_forward=False)
dsr = deflated_sharpe_ratio(
    obs_sr=bt["net_sharpe"],
    n_trials=30,
    n_obs=len(bt["net_returns"]),
    skew=0.0,
    kurtosis=3.0,
    return_details=True,
)
pbo = probability_backtest_overfitting(bt["net_returns"], n_groups=8)
```

DSR answers: “Given how hard we searched and the return distribution, how impressive is this Sharpe?”  
PBO answers: “How likely is the selected strategy to disappoint out of sample?”

Neither metric alone is an approval stamp.

---

## Stability

`SignalEvaluator` (used by `evaluate`) reports rolling / windowed stability using `research.stability_window` (default 60), `stability_step`, and `stability_min_obs` from Hydra. Unstable IC paths should block or delay promotion even when full-sample IC looks strong.

---

## Approve gates (validation-related)

Before `AlphaResearchEngine.approve`:

| Gate | Requirement |
|------|-------------|
| Hypothesis | Non-empty, ≥ `min_hypothesis_chars` |
| Not Sharpe-only | Reason/evidence must extend beyond Sharpe keywords when `allow_sharpe_only_approval=false` |
| Evidence | Report diagnostics must include validation keys (`significance`, `bootstrap`, `permutation`, `deflated_sharpe`, `pbo`, `validate`, `evaluate`, …) |
| Multiple testing | Prefer BH-FDR / Holm-adjusted significance at the session trial budget |

```python
from iqrp.app.alpha import ApprovalError

try:
    eng.approve(rec.experiment_id, reason="sharpe looks good")
except ApprovalError as e:
    print(e)  # refused: Sharpe-only / missing evidence
```

Recommended approval reason cites IC significance **after** MT adjustment, DSR/PBO, decay/capacity notes, and the economic hypothesis — not a single Sharpe number.

---

## Standalone evaluate report

```python
from iqrp.app.alpha.research.evaluator import SignalEvaluator
from iqrp.app.alpha.base.signal_result import SignalStatus

ev = SignalEvaluator(horizons=(1, 2, 5, 10), stability_window=60)
report = ev.evaluate(signal, returns, definition=defn, status=SignalStatus.RESEARCHING)
assert "Statistical significance alone ≠ alpha" in " ".join(report.warnings)
```

Warnings on the report intentionally restate architectural rules so downstream consumers cannot treat the report as a trading green light.
