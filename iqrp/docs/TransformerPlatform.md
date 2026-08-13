# Institutional Time-Series Transformer Forecasting Platform

Production transformer forecasting for IQRP. All models inherit from the Forecasting Framework (`ForecastModel` → `TransformerForecastModel`).

## Location

`iqrp/app/forecasting/transformers/`

## Models

| Name | Architecture |
|------|--------------|
| `tft` | Temporal Fusion Transformer |
| `informer` | Informer (ProbSparse / sparse attention) |
| `autoformer` | Autoformer |
| `fedformer` | FEDformer |
| `patchtst` | PatchTST |
| `crossformer` | Crossformer |
| `timesnet` | TimesNet |
| `itransformer` | iTransformer |
| `timemixer` | TimeMixer |
| `tide` | TiDE |
| `moe_transformer` | Mixture-of-Experts Transformer |

## Quick start

```python
from iqrp.app.forecasting.transformers import (
    TransformerSettings, TransformerOrchestrator, create_transformer_model,
)

model = create_transformer_model("patchtst")
model.fit(frame, feature_columns=["f0", "f1", "f2"], target_column="target")
pred = model.predict(frame)
fc = model.forecast(frame, horizon=8)
attn = model.attention(frame)
emb = model.embeddings(frame)
```

## API

`fit` · `partial_fit` · `predict` · `predict_proba` · `forecast` · `forecast_interval` · `attention` · `embeddings` · `evaluate` · `explain` · `export_onnx` · `diagnostics` · `save` / `load`

## Configuration

Hydra: `iqrp/configs/forecasting/transformers/default.yaml`

```python
TransformerSettings.from_hydra(overrides=["architecture.lookback=128", "architecture.attention_type=flash"])
```

## Integrations

- Feature Store / validated features as `feature_columns`
- Regime Intelligence via regime tokens / embeddings / MoE experts
- Volatility, tree, neural and statistical forecasts as input columns
- Probability Engine distributions for intervals / NLL
- Simulation Engine long-range synthetic series for validation
