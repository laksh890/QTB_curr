# State Space Framework

Institutional-grade, algorithm-agnostic latent-state modeling layer for IQRP.

**Scope:** framework + pluggable contract only. Production Markov / HMM /
Bayesian switching / Kalman / particle / DLM / Gaussian SSM detectors plug in
later without changing downstream research code.

## Location

`iqrp/app/state_space/`

```
app/state_space/
  base/           # contracts, latent state, results, registry, probability utils
  filtering/      # forward / backward filters
  smoothing/      # fixed-interval / fixed-lag smoothers
  forecasting/    # multi-step forecasts + uncertainty
  evaluation/     # metrics + diagnostics
  storage/        # Parquet + JSON + DuckDB
  visualization/  # SVG charts
  models/         # mock_discrete_ssm (framework validation only)
```

## Configuration

Hydra defaults: `iqrp/configs/state_space/default.yaml`

```python
from iqrp.app.state_space import StateSpaceSettings

settings = StateSpaceSettings.from_hydra(overrides=["forecasting.default_horizon=10"])
```

No hardcoded state counts, horizons, or dimensions — all come from settings /
model metadata.

## Core contract

Every algorithm implements `StateSpaceModel`:

| Method | Role |
|--------|------|
| `fit` | Fit on Polars / NumPy observations |
| `filter` | Forward filter → `FilterResult` |
| `smooth` | Fixed-interval / fixed-lag → `SmootherResult` |
| `predict` | Hard state IDs `(T,)` |
| `predict_proba` | Soft state probs `(T, K)` |
| `forecast` | Multi-step `ForecastResult` via `P^h` |
| `sample` | Simulate `(states, observations)` |
| `log_likelihood` | Marginal sequence log-likelihood |
| `save` / `load` | Artifact serialization |
| `evaluate` | AIC / BIC / accuracy / stability |

Supporting contracts:

- `TransitionModel` — `transition_matrix`, `sample_next_state`, `transition_probability`
- `ObservationModel` — `emission_probability`, `sample_observation`, `expected_observation`

## Workflow

```python
from iqrp.app.state_space import get_registry, StateStore

model = get_registry().create("mock_discrete_ssm", n_states=3)
model.fit(frame)

filt = model.filter(frame)
smooth = model.smooth(frame)
fc = model.forecast(frame, horizon=5)
print(filt.log_likelihood, fc.most_likely_path)

store = StateStore()
store.write_filter_result(
    filt,
    model_name=model.meta.name,
    version=model.meta.version,
    forecast=fc,
)
```

## Math engine dependency

All probability / matrix / stability operations go through `iqrp.app.math`:

- `logsumexp`, `stable_softmax` — scaled forward / backward
- `n_step_transition` — multi-step forecast matrix powers
- `normalize_rows`, `empirical_transition`, `simulate_markov`
- `aic` / `bic`, entropy / cross-entropy

## Registry

```python
from iqrp.app.state_space import register_state_space_model, StateSpaceModel

@register_state_space_model
class MyHMM(StateSpaceModel):
    meta = ...
```

Downstream code must depend only on `StateSpaceModel`.
