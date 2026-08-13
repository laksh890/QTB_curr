# Variational Inference

Mean-field VBEM for Bayesian HMM emissions and transitions.

## Factorization

\[
q(A)\, q(\pi)\, q(\theta)\, q(z)
\]

- \(q(z)\): forward–backward responsibilities  
- \(q(A)\), \(q(\pi)\): Dirichlet with expected transition counts  
- \(q(\theta)\): conjugate Gaussian / Inverse-Gamma (or Wishart) moments  

## Algorithm (`variational.py`)

1. E-step: smoothed state probabilities  
2. M-step: update variational Dirichlet / emission moments  
3. Track ELBO proxy (observed-data log-likelihood)  
4. Stop on `|ΔELBO| < tol` or `max_iter`  
5. Draw approximate posterior samples from updated conjugate factors  

## Configuration

```yaml
inference:
  algorithm: variational
variational:
  max_iter: 100
  tol: 1.0e-4
```

## Model comparison

Even under VI, `compare_models` reports WAIC / LOO proxies and a harmonic-mean marginal likelihood estimate for ranking \(K\).
