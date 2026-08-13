# N-BEATS

Neural Basis Expansion Analysis for interpretable time-series forecasting via stacked residual blocks with backcast / forecast branches.

## Usage

```python
from iqrp.app.forecasting.neural import create_neural_model

model = create_neural_model("nbeats")
model.fit(frame, feature_columns=cols)
fc = model.forecast(frame, horizon=5)
```

## Notes

Operates on flattened lookback × features. Supports quantile and distribution heads. Registered as `nbeats` in the Forecasting Framework registry.
