# SHAP Explainability

```python
sv = model.shap_values(frame)                 # (n, p)
ix = shap_interaction_values(estimator, X)    # (n, p, p)
grid, pdp = partial_dependence(estimator, X, 0)
grid, ice = ice_curves(estimator, X, 0)
paths = decision_paths(estimator, X)
```

Uses the `shap` library TreeExplainer when available; otherwise a KernelSHAP-style interventional sampler in `explainability/importance.py`.

`model.explain(frame, method="shap")` returns an `ExplanationResult` with mean |SHAP| importances and full attribution matrix.
