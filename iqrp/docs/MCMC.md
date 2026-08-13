# MCMC

## Gibbs sampling

1. **FFBS** — sample latent path \(z\) given current parameters  
2. **Transitions** — Dirichlet–Multinomial update  
3. **Emissions** — Normal–Inverse-Gamma (diag) or Normal–Wishart (full)  
4. Repeat; discard burn-in; thin

Multi-chain runs use a thread pool (`inference.n_jobs`). Numba accelerates FFBS when available.

## Metropolis–Hastings

Proposes means, log-variances, and soft-max transitions; accepts with complete-data log-joint ratio. Latent states refreshed via FFBS each iteration.

## Hamiltonian Monte Carlo

Leapfrog integrator on emission means with analytic gradient of the complete-data Gaussian likelihood; transitions/covariances refreshed conjugately.

## Convergence diagnostics (`convergence.py`)

- Gelman–Rubin \(\hat{R}\)
- Effective sample size (ESS)
- Autocorrelation
- Burn-in suggestion
- Acceptance rate

Exposed through `model.diagnostics()`.
