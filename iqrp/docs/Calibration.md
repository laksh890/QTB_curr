# Calibration

Probability calibration for ensemble outputs.

## Methods

| Method | Description |
|--------|-------------|
| `temperature` | Single temperature on log-probabilities |
| `platt` | Logistic scaling of confidence |
| `isotonic` | PAVA isotonic on confidence |
| `dirichlet` | Diagonal class reliability matrix |
| `none` | Passthrough |

## Metrics

- Expected Calibration Error (`expected_calibration_error`)
- Brier score (`brier_score`)

## Location

`iqrp/app/regimes/ensemble/calibration.py`

```python
model.calibrate(frame, true_states)
```
