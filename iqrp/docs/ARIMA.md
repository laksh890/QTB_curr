# ARIMA

`ARIMAModel` (`name="arima"`) implements ARIMA(p,d,q) via:

1. Automatic / configured differencing order `d` (ADF-based)
2. Conditional sum-of-squares ARMA on the differenced series
3. Forecast integration back to levels

## Usage

```python
from iqrp.app.forecasting.statistical.arima import ARIMAModel

model = ARIMAModel(p=1, d=1, q=1)
model.fit(frame, target_column="target")
fc = model.forecast(frame, horizon=5)
```

Auto order:

```python
from iqrp.app.forecasting.statistical import StatisticalTrainer
model, res = StatisticalTrainer().auto_arima(frame)
```

## Related

- AR / MA / ARMA are nested special cases
- SARIMA adds seasonal `(P,D,Q)s` structure
