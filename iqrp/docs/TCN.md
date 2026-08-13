# Temporal Convolutional Network (TCN)

Causal dilated convolutions for long-range temporal dependencies with parallelizable training.

## Usage

```python
from iqrp.app.forecasting.neural import create_neural_model, NeuralSettings

settings = NeuralSettings.from_mapping({
    "architecture": {"kernel_size": 3, "num_layers": 3, "hidden_size": 64},
})
model = create_neural_model("tcn", settings=settings)
model.fit(frame, feature_columns=cols)
```

## Notes

Residual temporal blocks with exponentially growing dilations. Suitable for short- and mid-horizon return / volatility proxy forecasting when latency matters.
