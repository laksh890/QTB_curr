# GRU Forecasting

Gated Recurrent Unit models for efficient sequence forecasting.

## Variants

- `gru` — standard / multi-layer GRU
- `stacked_gru` — deeper stacked GRU

## Usage

```python
from iqrp.app.forecasting.neural import create_neural_model

model = create_neural_model("gru")
model.fit(frame, feature_columns=cols)
pred = model.predict(frame)
```

## Notes

Supports bidirectional stacking via architecture settings. Shares the common `NeuralForecastModel` training stack (AMP, schedulers, early stopping, online fine-tuning).
