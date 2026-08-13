# Concept Drift

Detect distribution shift and trigger retraining.

## Signals

| Signal | Statistic |
|--------|-----------|
| Feature drift | Population Stability Index (PSI) |
| Prediction drift | Kolmogorov–Smirnov |
| Target drift | KS on labels |
| Covariate shift | Mean feature PSI |
| Performance degradation | Relative MAE increase |

## Retraining modes

`scheduled` · `performance` · `drift` · `rolling` · `none`

Supports warm-start (`partial_fit`) and checkpoint recovery.

## Usage

```python
report = engine.detect_drift(frame)
decision = engine.retrain(frame)  # respects RetrainConfig + drift
```
