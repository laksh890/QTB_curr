# Priors

Configured under `priors:` in Hydra / `PriorsConfig`.

## Supported families

| Prior | Use | Constructor |
|-------|-----|-------------|
| Dirichlet | Transitions / initial | `dirichlet_prior` |
| Beta | Scalar persistence / probabilities | `beta_prior` |
| Gamma | Rate / shape auxiliaries | `gamma_prior` |
| Inverse-Gamma | Diagonal emission variances | `inverse_gamma_prior` |
| Wishart | Precision for full covariance | `wishart_prior` |
| Normal | Univariate means | `normal_prior` |
| Multivariate Normal | Vector means | `mvn_prior` |
| User-defined | Arbitrary log-density + sampler | `UserDefinedPrior` |

## ModelPriors bundle

```python
from iqrp.app.regimes.bayesian.priors import ModelPriors
from iqrp.app.regimes.bayesian.config import BayesianSettings

s = BayesianSettings.default()
priors = ModelPriors.from_config(s.priors, n_states=3, n_features=1)
```

Defaults:

- `transition_alpha`, `initial_alpha` — Dirichlet concentration
- `mean_prior_location`, `mean_prior_strength` — Normal mean prior
- `invgamma_shape`, `invgamma_scale` — variance prior
- `wishart_df`, `wishart_scale` — full-covariance precision prior
