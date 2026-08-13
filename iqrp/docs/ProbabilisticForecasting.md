# Probabilistic Neural Forecasting

Quantile, distribution and uncertainty tooling for the Neural Forecasting Platform.

## Capabilities

- Quantile prediction (`task.type=quantile`, pinball / quantile loss)
- Distribution forecast (`gaussian_nll` / `student_t_nll`, DeepAR)
- Prediction intervals from quantiles or residual scales
- Aleatoric uncertainty from predictive sigma
- Epistemic uncertainty via MC dropout
- Total uncertainty aggregation

## Usage

```python
from iqrp.app.forecasting.neural import create_neural_model, NeuralSettings

settings = NeuralSettings.from_mapping({
    "task": {"type": "quantile", "quantile_alphas": [0.1, 0.5, 0.9]},
    "train": {"loss": "quantile"},
    "probabilistic": {"enabled": true},
})
model = create_neural_model("lstm", settings=settings)
model.fit(frame, feature_columns=cols)
intervals = model.forecast_interval(frame, horizon=5)
```

## Modules

- `probabilistic/distributions.py`
- `probabilistic/quantiles.py`
- `probabilistic/uncertainty.py`
