# Covariance Models

Supported parameterizations (`covariance.type`):

| Type | Storage | Free params per / overall |
|------|---------|---------------------------|
| `full` | `(K, D, D)` | \(K\cdot D(D+1)/2\) |
| `diag` | `(K, D)` | \(K\cdot D\) |
| `tied` | `(D, D)` | \(D(D+1)/2\) shared |
| `spherical` | `(K,)` | \(K\) |

Regularization: `covariance.reg_covar` added to diagonals / variances for numerical stability.

Helpers in `covariance.py`:

- `estimate_covariances`
- `expand_covariance` → always `(K, D, D)`
- `n_covariance_params` for AIC/BIC counting
