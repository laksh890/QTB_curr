# N-HiTS

Neural Hierarchical Interpolation for Time Series — multi-rate pooling blocks for multi-horizon forecasts.

## Usage

```python
from iqrp.app.forecasting.neural import create_neural_model

model = create_neural_model("nhits")
model.fit(frame, feature_columns=cols)
fc = model.forecast(frame, horizon=10)
```

## Notes

Hierarchical interpolation improves long-horizon stability relative to plain MLP / RNN baselines on seasonal financial proxies.
