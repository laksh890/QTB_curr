# Signal Ensemble

Combination and diversification of research signals: quality-aware weighting (not Sharpe-only), correlation analysis, redundancy detection, and hierarchical clustering.

**Package:** `iqrp.app.alpha.ensemble`  
**Engine entry:** `AlphaResearchEngine.compare`  
**Parent:** [AlphaResearch](AlphaResearch.md) · Related: [SignalRanking](SignalRanking.md) · [SignalCapacity](SignalCapacity.md)

---

## Governance

- Historical Sharpe alone cannot approve — and must not dominate ensemble weights
- Statistical significance alone ≠ alpha
- Ensembles are still research candidates until lifecycle-approved as their own experiment

Default composite score weights (`DEFAULT_SCORE_WEIGHTS`):

| Component | Weight | Notes |
|-----------|--------|-------|
| `ic` | 0.30 | Predictive strength |
| `stability` | 0.20 | Path consistency |
| `capacity` | 0.15 | Deployable AUM quality |
| `decay` | 0.15 | Inverted (faster decay → lower score) |
| `corr_penalty` | 0.10 | Inverted (redundancy → lower score) |
| `uncertainty` | 0.05 | Inverted |
| `sharpe` | 0.05 | **Capped** — never dominant |

---

## Combination and weighting

```python
import numpy as np
from iqrp.app.alpha.ensemble.signal_combination import (
    combine_signals,
    combine_from_metrics,
    rank_average_combine,
    majority_sign_combine,
)
from iqrp.app.alpha.ensemble.weighting import (
    compute_ensemble_weights,
    equal_weights,
    signal_quality_score,
)

rng = np.random.default_rng(1)
signals = {
    "mom": rng.normal(size=300),
    "mr": rng.normal(size=300),
    "alt": rng.normal(size=300),
}

# Equal weight
combo_eq = combine_signals(signals)  # normalize=True by default

# Explicit weights
combo_w = combine_signals(signals, weights={"mom": 0.5, "mr": 0.3, "alt": 0.2})

# Quality-aware (composite — not Sharpe-only)
metrics = {
    "mom": {"ic": 0.05, "stability": 0.7, "capacity": 0.8, "decay": 0.2, "sharpe": 1.4},
    "mr": {"ic": 0.03, "stability": 0.6, "capacity": 0.9, "decay": 0.4, "sharpe": 2.0},
    "alt": {"ic": 0.04, "stability": 0.5, "capacity": 0.4, "decay": 0.3, "sharpe": 0.9},
}
weights = compute_ensemble_weights(metrics, method="composite")
combo, w = combine_from_metrics(signals, metrics, method="composite")

# Other schemes
rank_avg = rank_average_combine(signals)
vote = majority_sign_combine(signals)
```

### Weight methods

`compute_ensemble_weights(..., method=)` accepts:

| Method | Behavior |
|--------|----------|
| `equal` | 1/N |
| `ic` | Proportional to IC quality |
| `risk_adj` | IC / uncertainty style |
| `corr_adj` | Down-weight correlated members |
| `regime` | Regime-conditional weights when metrics supply regime scores |
| `dynamic` | Time-varying when metrics include paths |
| `composite` | **Default research choice** — multi-factor with capped Sharpe |

```python
score = signal_quality_score(metrics["mom"])  # scalar quality in [~0, 1]
eq = equal_weights(["mom", "mr", "alt"])
```

---

## Correlation

```python
from iqrp.app.alpha import AlphaResearchEngine
from iqrp.app.alpha.ensemble.correlation import signal_correlation_matrix

eng = AlphaResearchEngine()
cmp = eng.compare(signals, returns=rng.normal(0, 0.01, 300))
# cmp["correlation"], cmp["redundancy"], cmp["signal_ic"]

corr = signal_correlation_matrix(signals, kind="prediction")
# kinds: return | position | prediction | ic | drawdown
# corr["matrix"], corr["labels"]
```

Use prediction/IC correlation for research redundancy; position/return correlation for portfolio clash detection.

---

## Redundancy

```python
from iqrp.app.alpha.ensemble.redundancy import (
    redundancy_report,
    find_high_correlation_pairs,
    detect_nested_signals,
)

red = redundancy_report(signals)
pairs = find_high_correlation_pairs(corr, threshold=0.85)
nested = detect_nested_signals(signals, r2_threshold=0.95)
```

Typical actions when redundancy is high:

1. Keep the member with better hypothesis + capacity + stability
2. Residualize or neutralize the weaker variant
3. Cluster and pick representatives (below)
4. Do not “diversify” by averaging near-duplicates and claiming lower risk

---

## Clustering

```python
from iqrp.app.alpha.ensemble.clustering import (
    hierarchical_correlation_clusters,
    correlation_distance,
)

dist = correlation_distance(corr["matrix"])
clusters = hierarchical_correlation_clusters(corr, threshold=0.5)
# cluster labels / members for representative selection
```

Distance transform: `sqrt(0.5 * (1 - ρ))`. Cluster representatives should still pass individual validation before any ensemble APPROVED experiment.

---

## Research pattern

```python
# 1) Compare book
report = eng.compare(signals)

# 2) Drop redundant names
keep = {k: v for k, v in signals.items() if k not in {"mr"}}  # example

# 3) Build ensemble candidate with documented hypothesis
from iqrp.app.alpha import SignalDefinition

ens = combine_from_metrics(keep, {k: metrics[k] for k in keep})[0]
defn = SignalDefinition(
    name="ens_mom_alt",
    version="0.1.0",
    formula="composite_weight(mom, alt)",
    features=("mom", "alt"),
    lookback=20,
    horizon=1,
    universe="default",
    frequency="1d",
    direction="long_short",
    expected_relationship="positive",
    economic_hypothesis=(
        "Diversified underreaction and alternative slow-info channels with "
        "low prediction correlation; weights favor IC/stability/capacity, "
        "not historical Sharpe."
    ),
    owner="research",
    signal_type="custom",
    tags=("ensemble", "candidate"),
)
rec = eng.register(defn, signal=ens)
```

Ranking triage: `eng.rank(candidates)` — still not approval. Promote the ensemble only after evaluate/validate/capacity like any other signal ([SignalLifecycle](SignalLifecycle.md)).
