# Extended Kalman Filter

First-order nonlinear Kalman filtering for IQRP.

## Location

`iqrp/app/regimes/kalman/ekf.py`

## Model

Nonlinear process / measurement:

\[
x_{t+1} = f(x_t) + w_t,\quad z_t = h(x_t) + v_t
\]

Jacobians \(F_t = \partial f / \partial x\), \(H_t = \partial h / \partial x\) may be
user-supplied (`f_jac`, `h_jac` on `LinearGaussianSSM`) or estimated by
`numerical_jacobian`.

## Usage

```python
from iqrp.app.regimes.kalman import KalmanFilterModel, KalmanSettings

settings = KalmanSettings.from_mapping({
    **KalmanSettings.default().model_dump(),
    "filter_type": "ekf",
    "application": "volatility",
})
model = KalmanFilterModel(settings=settings)
model.fit(squared_returns)
```

The `volatility` template uses an AR(1) log-variance state with exponential
observation map — the default EKF financial application.
