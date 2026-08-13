# Hidden Markov Model Engine

Production HMM for latent market-regime detection.

**Scope:** first-order discrete-time HMMs with discrete or Gaussian emissions.
Integrates with the State Space Framework (`HiddenMarkovModel`) and Regimes
layer (`HMMRegimeModel`). Uses the Probability / State-Space engines for
scaled forward-backward and matrix-power forecasts.

## Location

`iqrp/app/regimes/hmm/`

## Configuration

`iqrp/configs/hmm/default.yaml`

```python
from iqrp.app.regimes.hmm import HMMSettings, HiddenMarkovModel

settings = HMMSettings.from_hydra(overrides=["n_states=3", "emission.covariance_type=diag"])
model = HiddenMarkovModel(settings=settings)
model.fit(returns)
print(model.decode(returns)[:10], model.log_likelihood(returns))
```

## API

`fit`, `partial_fit`, `predict` / `decode` (Viterbi), `predict_proba` (smoothed),
`forward`, `backward`, `smooth`, `forecast`, `log_likelihood`, `aic`, `bic`,
`save` / `load`, `evaluate`, `diagnostics`, `select_model`.

## Emission families

- Discrete categorical
- Univariate / multivariate Gaussian (`diag` or `full` covariance)

## Training

Baum–Welch EM with KMeans / random / uniform initialization, multiple restarts,
parallel fitting (`training.n_jobs`), convergence tolerance, and warm-start
`partial_fit` for online updates.
