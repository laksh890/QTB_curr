# Feature Importance

```python
model.feature_importance(kind="gain")       # builtin / gain
model.feature_importance(kind="split")      # rank proxy
model.feature_importance(kind="permutation")
model.feature_importance(kind="shap")
```

## Selection methods

Configured under `feature_selection`:

- `rfe` / `permutation` / `shap` — importance ranking
- `mutual_info` — discrete MI
- `correlation` — pairwise filter
- `boruta` — shadow-feature thresholding

## Stability

`diagnostics().feature_stability` reports bootstrap stability of importances (0–1).
