# Institutional Forecasting Framework

Production-grade common infrastructure for every forecasting algorithm in IQRP.

## Purpose

This framework standardizes **training, prediction, evaluation, deployment, serialization, benchmarking, explainability, and online learning**. All future models (AR, ARIMA, SARIMA, VAR, Prophet, GARCH, LSTM, GRU, Transformer, TFT, XGBoost, LightGBM, CatBoost, Neural ODE, N-BEATS, DeepAR, …) must inherit from `ForecastModel` and register via `@register_forecast_model`.

Downstream modules — risk, portfolio, execution, live trading — should consume forecasts only through this framework’s objects (`Forecast`, `Prediction`, `PredictionInterval`).

## Package layout

```
iqrp/app/forecasting/
  base/           # ForecastModel, Forecast, Prediction, Trainer, Evaluator, Registry
  orchestration/  # Pipeline, Scheduler, Inference
  preprocessing/  # Scaling, encoding, windowing, feature selection
  postprocessing/ # Calibration, intervals, uncertainty
  explainability/ # Importance, attribution interfaces
  diagnostics/    # Residuals, drift, calibration reports
  serialization/  # JSON (+ NPZ) persistence
  visualization.py
  models/mock.py  # Framework stub (not a production alpha)
  config.py
```

## Common model API

Every model implements:

| Method | Role |
|--------|------|
| `fit` / `partial_fit` | Batch and online training |
| `predict` / `predict_proba` | In-sample / probabilistic outputs |
| `forecast` / `forecast_interval` | Multi-step path + intervals |
| `evaluate` / `explain` | Metrics and attributions |
| `save` / `load` | Persistence |
| `checkpoint` / `restore_checkpoint` | Warm-start recovery |

## Inputs

- Validated features / feature store frames
- Labels / targets
- Regime Intelligence outputs (`regime` column)
- Market data and custom numeric/categorical features

## Configuration

Hydra YAML: `iqrp/configs/forecasting/default.yaml`

```python
from iqrp.app.forecasting import ForecastingSettings, ForecastingPipeline

settings = ForecastingSettings.from_hydra(overrides=["inference.default_horizon=10"])
pipe = ForecastingPipeline(settings=settings, model_name="mock")
```

## Integration

- **Feature Platform** — frames with selected feature columns
- **Label Platform** — target columns / horizons
- **Regime Intelligence** — optional `regime_column`
- **Probability Engine** — softmax / entropy for uncertainty
- **Simulation Engine** — synthetic frames for validation
- **Serialization Framework** — `ForecastSerializer`

## Extending

```python
from iqrp.app.forecasting import ForecastModel, ForecastModelMeta, register_forecast_model

@register_forecast_model
class MyModel(ForecastModel):
    meta = ForecastModelMeta(
        name="my_model",
        version="0.1.0",
        description="...",
        algorithm_family="classical",
    )
    # implement fit, predict, forecast, _algorithm_state, _load_algorithm_state
```
