# Bayesian Regime Switching Engine

Production Bayesian regime detection with posterior uncertainty over both latent states and model parameters.

## Models

- **Bayesian HMM** (`model_type: bayesian_hmm`)
- **Bayesian Markov Switching** (`model_type: bayesian_markov_switching`)
- Gaussian and multivariate Gaussian emissions (diagonal or full covariance)

## Quick start

```python
import numpy as np
from iqrp.app.regimes.bayesian import BayesianRegimeSwitchingModel, BayesianSettings

settings = BayesianSettings.from_hydra(overrides=[
    "n_states=2",
    "inference.algorithm=gibbs",
    "inference.n_samples=100",
    "inference.burn_in=25",
    "inference.n_chains=2",
])
model = BayesianRegimeSwitchingModel(n_states=2, settings=settings, random_seed=0)
model.fit(y)  # y: (T, D) ndarray or Polars frame

states = model.predict(y)
proba = model.predict_proba(y)
post = model.posterior()
ci = model.credible_intervals("means", level=0.95)
fc = model.forecast(y, horizon=5)
```

## API

| Method | Role |
|--------|------|
| `fit` / `partial_fit` | Posterior inference (batch / online warm-start) |
| `predict` / `predict_proba` | MAP states / posterior state probabilities |
| `posterior` / `sample_posterior` | Posterior object / draw dictionaries |
| `posterior_predictive` | Simulated observations from posterior |
| `credible_intervals` | Parameter credible intervals |
| `forecast` | N-step posterior predictive regime forecast |
| `compare_models` | WAIC / LOO / marginal likelihood selection |
| `diagnostics` | Convergence + uncertainty report |
| `save` / `load` | JSON + NPZ checkpoint |

## Integration

Registered as State Space model `bayesian_regime` and Regime model `bayesian_regime`.

```python
import iqrp.app.regimes.bayesian  # noqa: F401
from iqrp.app.state_space import get_registry
assert "bayesian_regime" in get_registry().list_names()
```

## Configuration

Hydra defaults live in `iqrp/configs/bayesian/default.yaml`. Select inference via:

```yaml
inference:
  algorithm: gibbs  # gibbs | metropolis | hmc | variational
```
