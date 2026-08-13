# Gaussian Mixture Regime Detection

Production GMM engine for soft probabilistic market regime assignment.

## Quick start

```python
from iqrp.app.regimes.gmm import GaussianMixtureModel, GMMSettings

settings = GMMSettings.from_hydra(overrides=[
    "n_components=3",
    "covariance.type=full",
    "initialization.method=kmeans",
    "training.max_iter=100",
])
model = GaussianMixtureModel(n_components=3, settings=settings, random_seed=0)
model.fit(X)

labels = model.predict(X)
proba = model.predict_proba(X)   # soft regime membership
means = model.component_means()
covs = model.component_covariances()
```

## API

| Method | Role |
|--------|------|
| `fit` / `partial_fit` | Batch / online EM |
| `predict` / `predict_proba` | Hard / soft assignments |
| `score` / `log_likelihood` | Observed-data likelihood |
| `sample` | Draw component labels + observations |
| `component_means` / `component_covariances` | Parameter access |
| `cluster_statistics` | Occupancy + weight summary |
| `select_model` | AIC/BIC/ICL/CV component search |
| `outliers` / `diagnostics` | Density-based anomalies + reports |
| `save` / `load` | JSON + NPZ |

Registered State Space / Regime name: **`gmm`**.
