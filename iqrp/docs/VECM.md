# VECM

`VECMModel` estimates a Vector Error Correction Model using:

- Johansen-style eigenvalues for cointegration rank
- Engle–Granger bivariate check
- Reduced-form VECM dynamics for multi-step forecasting

```python
from iqrp.app.forecasting.statistical.vecm import VECMModel

model = VECMModel(lags=1)
model.fit(frame, feature_columns=["y0", "y1"])
print(model.cointegration_test())
fc = model.forecast(frame, horizon=5)
```
