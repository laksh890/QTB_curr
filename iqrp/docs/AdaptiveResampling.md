# Adaptive Resampling

ESS-triggered resampling for particle filters.

## Location

`iqrp/app/regimes/particle/resampling.py`, `adaptive.py`

## Schemes

| Method | Notes |
|--------|-------|
| Multinomial | Independent draws with replacement |
| Systematic | Single uniform offset (default) |
| Residual | Deterministic floor + residual multinomial |
| Stratified | One draw per stratum |

## Threshold

Hydra:

```yaml
resampling:
  adaptive: true
  ess_threshold: 0.5   # resample when ESS < 0.5 * N
  method: systematic
```

`adaptive_resample(cloud, ess_threshold=..., method=...)` returns `(cloud, did_resample)`.

Adaptive particle count (`suggest_n_particles` / `resize_cloud`) grows or shrinks \(N\) toward a target ESS fraction when `filter_type=adaptive`.
