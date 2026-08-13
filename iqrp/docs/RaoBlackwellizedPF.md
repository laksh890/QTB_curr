# Rao-Blackwellized Particle Filter

Marginalizes conditionally linear-Gaussian state components analytically.

## Location

`iqrp/app/regimes/particle/trainer.py` → `filter_rao_blackwellized`

## State factorization

- **Nonlinear** component: tracked by weighted particles (transition from `TransitionModel`)
- **Linear** component: per-particle Kalman filter with identity dynamics

Observation model:

\[
z_t \approx h_{\text{nl}}(x_t^{\text{nl}}) + H x_t^{\text{lin}} + v_t
\]

Particle weights use the innovation likelihood after the KF update.

## Hydra

```yaml
filter_type: rao_blackwellized
rao_blackwellized:
  n_linear: 1
  kalman_process_noise: 1.0e-3
  kalman_observation_noise: 1.0e-2
```

## When to use

Systems with a nonlinear trend / regime factor plus a linear residual level or factor loading — lower variance than a fully particle-discretized state of the same dimension.
