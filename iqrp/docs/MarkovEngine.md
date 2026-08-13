# Markov Chain Engine

Institutional discrete-time, time-homogeneous Markov chain for finite-state
regime modeling.

**Scope:** first concrete implementation of the State Space Framework. Also
exposes a `RegimeModel` adapter (`MarkovRegimeModel`) for the regimes layer.
No Bull/Bear assumptions — works for any finite `K`.

## Location

`iqrp/app/regimes/markov/`

```
markov/
  model.py          MarkovChainModel (StateSpace) + MarkovRegimeModel
  estimator.py      MLE / Bayesian / frequency / weighted / online
  transition.py     counts, Laplace, sparse, incremental updates
  forecast.py       1..N step via P^h
  persistence.py    durations, occupancy, switch rate
  stationary.py     π, ergodicity, mixing time
  trainer.py        fit / partial_fit orchestration
  evaluator.py      LL, AIC/BIC, accuracy, calibration
  diagnostics.py    entropy, rare states, low-sample warnings
  visualization.py  SVG charts
  serializer.py     JSON + npz
  state_mapper.py   label → state id
  config.py         Hydra settings
```

## Configuration

`iqrp/configs/markov/default.yaml`

```python
from iqrp.app.regimes.markov import MarkovSettings, MarkovChainModel

settings = MarkovSettings.from_hydra(overrides=["n_states=4", "estimation.method=bayesian"])
model = MarkovChainModel(settings=settings)
```

## Quick start

```python
import iqrp.app.regimes.markov  # registers with StateSpace + Regime registries
from iqrp.app.regimes.markov import MarkovChainModel

model = MarkovChainModel(n_states=3)
model.fit(state_ids)
print(model.transition_matrix())
print(model.stationary_distribution())
print(model.forecast(state_ids, horizon=5).most_likely_path)
model.partial_fit(new_states)  # online update
```

## State Space contract

Implements `StateSpaceModel`: `fit`, `filter`, `smooth`, `predict`,
`predict_proba`, `forecast`, `sample`, `log_likelihood`, `save`/`load`,
`evaluate`, plus `partial_fit`, `transition_matrix`, `stationary_distribution`,
`expected_duration`, `state_probabilities`.

## Regime adapter

`MarkovRegimeModel` registers as `markov_chain` on the regime registry for use
with existing regime services without modifying them (import the markov package
first).
