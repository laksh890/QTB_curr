# Probability Engine

Institutional probability primitives under `iqrp.app.math.probability`.

## Distributions

Gaussian, Multivariate Gaussian, Student-t, Bernoulli, Binomial, Poisson,
Exponential, Gamma, Beta, Dirichlet, Uniform, Laplace, LogNormal, Chi-square,
F, Weibull, Cauchy, and finite Mixtures.

```python
from iqrp.app.math.probability import gaussian, MixtureDistribution, get_distribution

dist = gaussian(0.0, 1.0)
print(dist.logpdf([0.0, 1.0]))
mix = MixtureDistribution([0.7, 0.3], [gaussian(0, 1), gaussian(3, 1)])
```

## Sampling

Random, weighted, importance, stratified, bootstrap, Monte Carlo, rejection,
and systematic resampling.

## Bayesian utilities

Bayes rule (linear + log-space), prior updates, evidence, posterior predictive.

## Likelihood

Log-likelihood, NLL, joint / conditional likelihood, Gaussian MLE, AIC/BIC,
numerical maximum likelihood.
