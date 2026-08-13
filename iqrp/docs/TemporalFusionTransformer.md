# Temporal Fusion Transformer (TFT)

Variable selection via gated residual networks, temporal self-attention and a decoder query path for multi-horizon forecasting.

```python
from iqrp.app.forecasting.transformers import create_transformer_model
model = create_transformer_model("tft")
model.fit(frame, feature_columns=cols)
```
