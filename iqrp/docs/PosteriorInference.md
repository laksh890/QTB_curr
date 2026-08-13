# Posterior Inference

## Posterior object

`Posterior` stores MCMC / VI draws (`ParameterDraw`) and exposes:

- `mean_transition`, `mean_initial`, `mean_means`, `mean_covars`
- `credible_intervals(parameter, level=0.95)`
- `posterior_state_probabilities()` — empirical \(P(Z_t\mid data)\)
- `state_occupancy()`
- `marginal_summary(parameter)`
- `scalar_ci(values)`

## Predictive

```python
y_rep = model.posterior_predictive(n_steps=100)
```

`posterior_predictive_observations` draws regime paths and emissions from each retained parameter draw.

## Credible intervals

```python
ci = model.credible_intervals("transition", level=0.9)
# keys: mean, low, high, level, parameter
```

## Joint vs marginal

Each `ParameterDraw` is a joint sample \((A,\pi,\mu,\Sigma,z)\). Marginal summaries are formed by stacking the relevant slice across draws.
