# LSTM Forecasting

Long Short-Term Memory sequence models for institutional forecasting.

## Variants

- `lstm` — standard / multi-layer LSTM
- `stacked_lstm` — deeper stacked LSTM (`num_layers=3`)
- `bidirectional_lstm` — BiLSTM over the lookback window

## Usage

```python
from iqrp.app.forecasting.neural import create_neural_model, NeuralSettings

settings = NeuralSettings.from_mapping({
    "architecture": {"lookback": 32, "horizon": 5, "hidden_size": 64, "num_layers": 2},
    "train": {"epochs": 30, "optimizer": "adamw", "loss": "mse"},
})
model = create_neural_model("lstm", settings=settings)
model.fit(frame, feature_columns=cols)
fc = model.forecast(frame, horizon=5)
```

## Notes

Positional encodings can be enabled inside `LSTMNet`. Supports regression, classification, quantile and Gaussian distribution heads via `task.type` / `probabilistic`.
