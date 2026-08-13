# Seq2Seq Forecasting

Encoder–decoder sequence-to-sequence models for multi-step forecasting.

## Variants

- `seq2seq` — encoder-decoder (attention optional via settings)
- `attention_seq2seq` — attention enabled by default

## Usage

```python
from iqrp.app.forecasting.neural import create_neural_model

model = create_neural_model("attention_seq2seq")
model.fit(frame, feature_columns=cols)
fc = model.forecast(frame, horizon=5)
# attention maps available on the module after forward: module.last_attention
```

## Components

- `encoder.py` — LSTM encoder
- `decoder.py` — LSTM decoder
- `attention.py` — additive / Bahdanau-style attention
- `model.py` — registered `Seq2SeqForecastModel`
