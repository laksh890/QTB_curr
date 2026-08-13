# Bayesian Inference

The Bayesian Regime Switching Engine estimates **posterior distributions** over transitions, emissions, and latent regimes rather than point MLEs.

## Joint model

\[
p(A,\pi,\theta,z \mid y) \propto p(y \mid z,\theta)\, p(z \mid A,\pi)\, p(A)\, p(\pi)\, p(\theta)
\]

- \(A\): transition matrix (Dirichlet row priors)
- \(\pi\): initial distribution (Dirichlet)
- \(\theta\): emission parameters (Normal–Inverse-Gamma / Wishart)
- \(z\): latent regime path

## Algorithms

| Algorithm | Module | Notes |
|-----------|--------|-------|
| Gibbs | `gibbs.py` | FFBS + conjugate updates; multi-chain parallel |
| Metropolis–Hastings | `metropolis.py` | Random-walk on means / log-variances / transitions |
| HMC | `hmc.py` | Leapfrog on emission means; Gibbs refresh elsewhere |
| Variational | `variational.py` | Mean-field VBEM + posterior draws from factors |

## Online updating

`partial_fit` supports warm-start Gibbs refinement, update frequency gating, and rolling windows via `online.*` settings.

## Checkpointing

When `inference.checkpoint_every > 0`, Gibbs writes NPZ checkpoints under `store_dir/checkpoints`.
