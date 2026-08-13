# Forecast Objects

## `Prediction`

Atomic point forecast: value, timestamp, horizon, probability, class id, regime, features used, model version, metadata.

## `PredictionInterval`

Lower / upper bounds with `level`, `kind` (`prediction` | `confidence`), optional midpoint.

## `QuantileForecast`

Map of quantile → value at a horizon.

## `DistributionForecast`

Mean, variance, optional samples, parametric `params`.

## `Forecast`

Canonical multi-horizon container consumed by downstream IQRP modules:

| Field | Description |
|-------|-------------|
| `values` | Point path `(H,)` |
| `horizon` | Forecast horizon |
| `timestamps` | Optional aligned timestamps |
| `intervals` / `confidence_intervals` | Uncertainty bands |
| `probabilities` | Class / state probabilities |
| `quantiles` / `distribution` | Rich uncertainty |
| `features_used` | Feature columns |
| `regime_used` | Regime Intelligence context |
| `model_name` / `model_version` | Provenance |
| `strategy` | `direct` \| `recursive` \| `sequence` \| `multi_step` |
| `metadata` | Extensible bag |

### Helpers

- `Forecast.from_values(...)`
- `point(step)` / `one_step()` / `n_step(n)`
- `path()` → `np.ndarray`
- `to_dict()` for serialization / logging
