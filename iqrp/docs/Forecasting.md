# Forecasting

Multi-step latent-state forecasts via matrix exponentiation.

## Location

`iqrp/app/state_space/forecasting/`

- `multi_step.py` — horizon-`h` distributions using math-engine `n_step_transition`
- `uncertainty.py` — entropy / HPD / max-probability bands

## Multi-step recursion

Given filtered occupancy `π_t` and row-stochastic `P`:

```
π_{t+h} = π_t P^h
```

`P^h` is computed exclusively through `iqrp.app.math.stochastic.markov_utils.n_step_transition`.

```python
from iqrp.app.state_space import MultiStepForecaster

fc = MultiStepForecaster(settings).forecast(pi_t, transition, horizon=10)
print(fc.expected_state, fc.most_likely_path, fc.confidence_interval)
```

## ForecastResult

| Field | Meaning |
|-------|---------|
| `horizon` | Steps ahead |
| `expected_state` | argmax of terminal distribution |
| `probability_distribution` | Terminal `π P^h` |
| `confidence_interval` | HPD probability mass bounds |
| `expected_duration` | Geometric sojourn times from `P_ii` |
| `step_distributions` | Row `h` is `π P^{h+1}` for `h=0..H-1` |

Horizon and confidence level come from Hydra (`forecasting.default_horizon`,
`forecasting.confidence_level`).

## Markov Chain Engine

For the discrete Markov specialization see `MarkovEngine.md` and
`iqrp/app/regimes/markov/forecast.py` (`MarkovForecaster`).
