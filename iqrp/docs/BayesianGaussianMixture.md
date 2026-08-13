# Bayesian Gaussian Mixture

Set `model_type: bayesian_gmm` to enable variational Bayesian EM.

## Priors

Configured under `bayesian:`:

- `weight_concentration_prior` — Dirichlet concentration on mixture weights
- `mean_precision_prior` — Normal mean shrinkage strength
- `covariance_prior_scale` — covariance regularization / prior scale

## Algorithm

1. E-step: responsibilities from current parameters  
2. Bayesian M-step: Dirichlet weight update + Normal-shrunk means + scaled covariances  

## Usage

```python
settings = GMMSettings.from_hydra(overrides=[
    "model_type=bayesian_gmm",
    "n_components=3",
    "bayesian.weight_concentration_prior=1.0",
])
model = GaussianMixtureModel(settings=settings)
model.fit(X)
```

Bayesian GMM tends to shrink unused components toward the prior, aiding automatic relevance determination when over-specified.
