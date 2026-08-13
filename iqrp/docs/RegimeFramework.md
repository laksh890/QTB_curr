# Market Regime Detection Framework

Institutional-grade, algorithm-agnostic regime detection layer for IQRP.

**Scope:** framework + pluggable contract only. Production Markov / HMM / GMM /
Kalman / particle / neural detectors plug in later without changing downstream
forecasting code.

## Location

`iqrp/app/regimes/`

```
app/regimes/
  base/           # contract, state, transition, probs, persistence, registry, eval
  services/       # detector, trainer, predictor, serializer
  storage/        # Parquet + DuckDB
  visualization/  # SVG charts
  models/         # mock_regime (framework validation only)
```

## Configuration

Hydra defaults: `iqrp/configs/regimes/default.yaml`

```python
from iqrp.app.regimes import RegimeSettings

settings = RegimeSettings.from_hydra(overrides=["detection.confidence_threshold=0.6"])
```

## Core contract

Every algorithm implements `RegimeModel`:

| Method | Role |
|--------|------|
| `fit` | Fit on a Polars frame |
| `predict` | Hard state IDs `(T,)` |
| `predict_proba` | Soft state probs `(T, K)` |
| `transition_matrix` | `K×K` row-stochastic matrix |
| `state_sequence` | Typed `RegimeState` list |
| `forecast` | 1..N step `RegimeForecast` |
| `save` / `load` | Artifact serialization |
| `evaluate` | Accuracy / LL / stability metrics |

Downstream code must depend only on `RegimeModel` / `RegimeDetector`.

## Workflow

```python
from iqrp.app.regimes import RegimeDetector

detector = RegimeDetector()
assert "mock_regime" in detector.available_models()

result = detector.detect(
    ohlcv,
    model_name="mock_regime",
    persist=True,
    exchange="binance",
    symbol="BTCUSDT",
    timeframe="1h",
    write_charts=True,
)

print(result.state_ids[:5])
print(result.forecast.most_likely_path)
print(result.persistence.expected_duration)
```

## First-class objects

- **RegimeState** — id, name, probability, confidence, persistence, times, features, model version
- **RegimeTransition** — previous/current state, probability, confidence, timestamp
- **RegimeForecast** — 1/N-step distribution, confidence intervals, expected duration
- **ProbabilityEngine** — state / transition / joint / conditional / forecast probabilities
- **PersistenceEngine** — durations, expected duration, persistence score, rolling persistence

## Registry

Models self-register via `@register_regime_model` and are discovered by name with
versioning / metadata / configuration through `RegimeModelMeta`.

```python
from iqrp.app.regimes import get_registry, ensure_regime_models_loaded

ensure_regime_models_loaded()
print(get_registry().list_names())
print(get_registry().describe("mock_regime").to_dict())
```

## Storage

`RegimeStore` writes Hive-style Parquet partitions:

`exchange=…/symbol=…/timeframe=…/model=…/version=…/`

Artifacts: `states.parquet`, `transition_matrix.parquet`, `probabilities.parquet`,
`forecast.json`, `metadata.json`, plus optional DuckDB views.

## Visualization

SVG outputs (no heavy plotting deps):

- Regime timeline
- Transition matrix heat map
- Rolling persistence
- State probability paths

## Evaluation

`RegimeEvaluator` reports prediction accuracy, transition accuracy, log-likelihood,
cross-entropy, state stability, and persistence stability (supervised or unsupervised).

## Future algorithms

Plug-in families (not implemented here):

- Markov Chains / Hidden Markov Models
- Bayesian Switching Models
- Gaussian Mixture Models
- Kalman / Particle Filters
- Neural Regime Models
