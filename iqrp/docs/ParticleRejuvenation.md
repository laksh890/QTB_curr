# Particle Rejuvenation

Restore particle diversity after resampling.

## Location

`iqrp/app/regimes/particle/rejuvenation.py`

## Methods

| Method | Behavior |
|--------|----------|
| `jitter` | Isotropic Gaussian noise |
| `covariance` | Noise along empirical posterior covariance |
| `adaptive` | Jitter scale inflated when ESS is low |
| `mcmc` | Random-walk Metropolis using observation likelihood |

## Hydra

```yaml
rejuvenation:
  enabled: true
  method: jitter
  scale: 0.05
  mcmc_steps: 1
```

Applied automatically after resampling in bootstrap / SIR / APF / adaptive runners when enabled.
