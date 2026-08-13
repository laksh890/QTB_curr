# Institutional Neural Forecasting Platform

Production neural sequence forecasting for IQRP. All models inherit from the Forecasting Framework (`ForecastModel` → `NeuralForecastModel`).

## Location

`iqrp/app/forecasting/neural/`

## Models

| Name | Architecture |
|------|--------------|
| `mlp` | Multi-Layer Perceptron |
| `lstm` | LSTM |
| `stacked_lstm` | Stacked LSTM |
| `bidirectional_lstm` | Bidirectional LSTM |
| `gru` | GRU |
| `stacked_gru` | Stacked GRU |
| `tcn` | Temporal Convolutional Network |
| `nbeats` | N-BEATS |
| `nhits` | N-HiTS |
| `deepar` | DeepAR |
| `seq2seq` | Encoder-Decoder Seq2Seq |
| `attention_seq2seq` | Attention Seq2Seq |

## Tasks

Regression · Classification · Probability · Quantile · Distribution · Multi-horizon · Multi-target sequence prediction

## Quick start

```python
from iqrp.app.forecasting.neural import (
    NeuralSettings, NeuralOrchestrator, create_neural_model,
)

model = create_neural_model("lstm")
model.fit(frame, feature_columns=["f0", "f1", "f2"], target_column="target")
pred = model.predict(frame)
fc = model.forecast(frame, horizon=5)
intervals = model.forecast_interval(frame, horizon=5)
expl = model.explain(frame, method="integrated_gradients")
```

## API

`fit` · `partial_fit` · `predict` · `predict_proba` · `forecast` · `forecast_interval` · `evaluate` · `explain` · `export_onnx` · `diagnostics` · `save` / `load`

## Configuration

Hydra: `iqrp/configs/forecasting/neural/default.yaml`

```python
NeuralSettings.from_hydra(overrides=["train.epochs=20", "architecture.lookback=48", "train.loss=huber"])
```

## Integrations

- Validated features / Feature Store as `feature_columns`
- Regime Intelligence (`feature` / `embedding` / `gating` / `separate` / `moe`)
- Volatility, statistical and tree forecasts as input columns
- Probability Engine distributions for intervals / NLL
- Simulation Engine synthetic nonlinear datasets for validation
- Serialization via Forecasting Framework `ForecastSerializer`
