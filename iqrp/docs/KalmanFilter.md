# Kalman Filter

Linear-Gaussian filtering for continuous latent-state estimation in IQRP.

## Location

`iqrp/app/regimes/kalman/`

| Module | Role |
|--------|------|
| `linear.py` | Batch linear KF |
| `prediction.py` | Predict / n-step forecast |
| `update.py` | Innovation, gain, Joseph update |
| `model.py` | State Space + Regime adapters |
| `initialization.py` | Financial SSM templates |

## State-space form

\[
x_{t+1} = F x_t + B u_t + w_t,\quad w_t \sim \mathcal{N}(0, Q)
\]
\[
z_t = H x_t + v_t,\quad v_t \sim \mathcal{N}(0, R)
\]

Supports time-varying \(H_t, F_t, Q_t, R_t\) and control inputs.

## API

```python
from iqrp.app.regimes.kalman import KalmanFilterModel, KalmanSettings

model = KalmanFilterModel(settings=KalmanSettings.from_hydra(
    overrides=["application=trend", "filter_type=linear"]
))
model.fit(observations)
filt = model.filter(observations)
sm = model.smooth(observations)
fc = model.forecast(observations, horizon=5)
x = model.state()
P = model.covariance()
```

Continuous means/covariances live in `FilterResult.metadata` and model accessors.
Soft bullish/bearish probabilities are exposed via `predict` / `predict_proba` for the
regime contract (`FilterResult` casts discrete states to `int64`).

## Applications

Built by `build_system()`: `trend`, `denoise`, `dynamic_beta`, `volatility`,
`spread`, `pairs`, `custom`.

## Integration

Registered as `"kalman"` on import (`@register_state_space_model` /
`@register_regime_model`). Hydra defaults: `iqrp/configs/kalman/default.yaml`.
