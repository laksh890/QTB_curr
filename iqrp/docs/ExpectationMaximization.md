# Expectation Maximization

Classical EM for GMMs lives in `em.py`, with E/M steps in `expectation.py` / `maximization.py`.

## Loop

1. **Initialize** weights, means, covariances (`random` / `kmeans` / `kmeans++` / `hierarchical` / `user`)  
2. **E-step** — responsibilities \(r_{ik} \propto \pi_k\,\mathcal{N}(x_i\mid\mu_k,\Sigma_k)\)  
3. **M-step** — update \(\pi,\mu,\Sigma\) from weighted sufficient statistics  
4. Monitor average log-likelihood; stop when `|Δℓ| < tol` or `max_iter`

## Features

- Multiple random restarts (parallel via thread pool)
- Early stopping
- Warm-start for online `partial_fit`
- Numba-ready Gaussian densities (optional)

## Convergence

`EMResult` exposes `history`, `n_iter`, `converged`, and final `log_likelihood`.
