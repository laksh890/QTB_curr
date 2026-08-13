# Forecasting Model Lifecycle

## Stages

1. **Discover / Register** — `@register_forecast_model` at import time; `ensure_forecast_models_loaded`
2. **Configure** — `ForecastingSettings` (Hydra)
3. **Preprocess** — scale, encode, select, window
4. **Train** — `ForecastTrainer.fit` / `partial_fit` with optional validation split
5. **Evaluate** — holdout, walk-forward, rolling, time-series CV
6. **Explain** — permutation / SHAP / IG / attention interfaces
7. **Diagnose** — residuals, drift, calibration, bias
8. **Deploy** — `ForecastingPipeline`, `StreamingInference`, `ForecastScheduler`
9. **Persist** — `save` / `load`, checkpoints for warm start
10. **Adapt** — online `partial_fit`, rolling retrain, checkpoint recovery

## Online learning

```python
from iqrp.app.forecasting import ForecastScheduler, ForecastingSettings

sched = ForecastScheduler(ForecastingSettings.from_mapping({
    "online": {"warm_start": True, "rolling_retrain_every": 50, "checkpoint_every": 25}
}))
sched.on_update(model, batch_frame, feature_columns=["f0"], target_column="target")
```

## Checkpoint recovery

```python
payload = model.checkpoint()
# ... failure ...
model.restore_checkpoint(payload)
```

## Registry metadata

The registry stores class meta, optional configs, and a training history (`TrainingMetadata`) for auditability.
