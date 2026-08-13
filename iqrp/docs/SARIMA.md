# SARIMA

`SARIMAModel` (`name="sarima"`) supports seasonal ARIMA `(p,d,q)(P,D,Q)s`.

- Seasonal period defaults to `order.seasonal_period` (12)
- `D` suggested via seasonal ACF + ADF when `identification.seasonal_detect=true`
- Forecasts recurse with seasonal AR terms then integrate seasonal / regular differences

```python
from iqrp.app.forecasting.statistical.sarima import SARIMAModel

model = SARIMAModel(seasonal_period=12)
model.fit(frame, target_column="target")
```
