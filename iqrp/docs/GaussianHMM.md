# Gaussian HMM

Gaussian emission specialization of the Institutional HMM Engine.

## Emission model

`GaussianEmissionModel` in `emissions.py`

| Covariance | Storage | Update |
|------------|---------|--------|
| `diag` | `(K, D)` variances | Weighted per-dimension variance |
| `full` | `(K, D, D)` SPD | Weighted outer products + jitter |

Log-densities are vectorized in NumPy with stable `slogdet` / fallback `pinv`.

## Initialization

KMeans (default), random, or uniform means; empirical cluster covariances with
`training.min_covar` floor.

## Validation

Recover means / transitions on synthetic data from
`HiddenRegimeSimulator` / `RegimeSwitchingSimulator`.
