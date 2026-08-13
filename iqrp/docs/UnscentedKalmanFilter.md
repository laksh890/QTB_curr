# Unscented Kalman Filter

Sigma-point Kalman filtering without analytic Jacobians.

## Location

`iqrp/app/regimes/kalman/ukf.py`

## Algorithm

1. Generate \(2n+1\) sigma points from \((x, P)\) with parameters
   \(\alpha, \beta, \kappa\) (Hydra `ukf` block).
2. Propagate through \(f\) / \(h\) (unscented transform).
3. Form weighted mean, covariance, and cross-covariance \(P_{xy}\).
4. Kalman gain \(K = P_{xy} S^{-1}\); Joseph-stable covariance update.

## Usage

```python
settings = KalmanSettings.from_hydra(overrides=["filter_type=ukf", "application=volatility"])
model = KalmanFilterModel(settings=settings).fit(y)
```

Helpers: `sigma_points`, `unscented_transform`, `filter_ukf`.
