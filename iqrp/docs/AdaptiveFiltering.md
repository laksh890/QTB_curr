# Adaptive Filtering

Online process / observation noise adaptation for Kalman filters.

## Location

`iqrp/app/regimes/kalman/adaptive.py`

## Mechanisms

| Mechanism | Description |
|-----------|-------------|
| Innovation monitoring | Rolling window of residuals |
| Covariance matching | Empirically match \(S\) to sample innov cov → update \(R\) |
| Process inflation | Inflate \(Q\) when Mahalanobis distance exceeds threshold |
| Batch EM-style | `adapt_noise_from_trace` + trainer iterations |

## Hydra

```yaml
filter_type: adaptive
adaptive:
  enabled: true
  window: 20
  process_adapt_rate: 0.05
  observation_adapt_rate: 0.05
  innovation_threshold: 3.0
```

## Usage

```python
model = KalmanFilterModel(settings=KalmanSettings.from_hydra(
    overrides=["filter_type=adaptive", "application=denoise"]
))
model.fit(noisy_prices)
diag = model.diagnostics(noisy_prices)
print(diag["noise_estimates"])
```

Streaming: `model.update(z_t)` for real-time correction with warm-started
\((x, P)\`; `partial_fit` batches streaming frames.
