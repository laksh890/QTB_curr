# DeepAR

Probabilistic autoregressive RNN forecasting with Gaussian (or Student-t) predictive distributions.

## Usage

```python
from iqrp.app.forecasting.neural import create_neural_model, NeuralSettings

settings = NeuralSettings.from_mapping({
    "task": {"type": "distribution"},
    "train": {"loss": "gaussian_nll"},
    "probabilistic": {"enabled": true, "distribution": "gaussian"},
})
model = create_neural_model("deepar", settings=settings)
model.fit(frame, feature_columns=cols)
fc = model.forecast(frame, horizon=5)
# intervals / quantiles from predictive sigma
```

## Notes

Default DeepAR construction prefers Gaussian NLL. Supports prediction intervals, aleatoric uncertainty and optional MC-dropout epistemic estimates.
