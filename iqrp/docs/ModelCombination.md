# Model Combination

Ensemble combination methods for unified regime posteriors.

## Methods

| Method | Behavior |
|--------|----------|
| `majority` | Hard vote → normalized counts |
| `weighted` | Weighted hard vote |
| `soft_voting` | Weighted average of probabilities (default) |
| `bma` | Softmax of log-evidence → soft voting |
| `stacking` | Linear meta-weights (vector or per-class) |
| `confidence` | Base weight × per-step max-proba |
| `dynamic` | Blend soft + confidence |
| `meta` | Select single best member by score |

## Location

`iqrp/app/regimes/ensemble/combiner.py` → `combine(...)`
