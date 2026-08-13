# Stochastic Models

Path generators registered in `iqrp.app.simulation.stochastic`.

| Name | Class | Family |
|------|-------|--------|
| `gbm` | GeometricBrownianMotion | diffusion |
| `abm` | ArithmeticBrownianMotion | diffusion |
| `ou` | OrnsteinUhlenbeck | mean_reversion |
| `merton_jump` | MertonJumpDiffusion | jump_diffusion |
| `jump_diffusion` | JumpDiffusion (Merton alias) | jump_diffusion |
| `heston` | HestonModel | stochastic_volatility |
| `variance_gamma` | VarianceGamma | levy |
| `cir` | CoxIngersollRoss | mean_reversion |
| `random_walk` | RandomWalk | discrete |

## Contract

Every generator implements:

```python
generate(n_steps, *, x0, dt, **params) -> PathResult
```

`PathResult` contains `prices`, `returns`, `volatility`, `drift`, and optional `latent` state.

## Noise innovations

Configured via `noise.distribution`:

- `gaussian`, `student_t`, `laplace`, `cauchy`, `uniform`, `mixture`

Multi-asset shocks are correlated through a Cholesky factor of the scenario
correlation matrix.

## Parameter sources

Dynamics parameters are Hydra-driven (`dynamics.*`) and overridable per `Scenario.parameters`.

Regime overlays replace per-step drift / volatility when `regime_enabled=True`.
