# Sequential Monte Carlo

Overview of SMC algorithms in the Institutional Particle Filter Engine.

## Algorithms

| Filter | Module entry | Description |
|--------|--------------|-------------|
| Bootstrap PF | `filter_bootstrap` | Prior proposal + adaptive resampling |
| SIS | `filter_sis` | Importance sampling, no resampling |
| SIR | `filter_sir` | Always resample each step |
| Auxiliary PF | `filter_auxiliary` | Pitt–Shephard look-ahead weights |
| Rao-Blackwellized | `filter_rao_blackwellized` | Particles + per-particle KF |
| Adaptive PF | `filter_adaptive` | ESS-driven N / proposal adaptation |

## Recursion

1. **Propose** \(x_t^{(i)} \sim q(x_t \mid x_{t-1}^{(i)}, z_t)\)
2. **Weight** \(\log w_t^{(i)} \gets \log w_{t-1}^{(i)} + \log p(z_t \mid x_t^{(i)}) + \log\) ratio
3. **Normalize** via stable softmax / logsumexp
4. **Resample** if ESS \(< \tau N\)
5. **Rejuvenate** (optional MCMC / jitter)

## Likelihoods

Gaussian, Student-t, Laplace (Hydra `likelihood`), plus custom callables.
