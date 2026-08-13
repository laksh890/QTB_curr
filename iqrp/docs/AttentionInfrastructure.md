# Attention Infrastructure

## Modules

- `multihead.py` — scaled dot-product multi-head attention
- `flash_attention.py` — SDPA / chunked memory-efficient attention
- `sparse_attention.py` — local window + global tokens
- `performer.py` — FAVOR+ style kernel attention
- `linear_attention.py` — ELU-feature linear attention
- `cross_asset_attention.py` — cross-sectional / multi-asset attention
- `temporal_attention.py` — temporal + hierarchical attention

## Factory

```python
from iqrp.app.forecasting.transformers.attention import build_attention
attn = build_attention("flash", d_model=64, n_heads=4)
```

Set `architecture.attention_type` in Hydra to `full`, `flash`, `sparse`, `linear`, `performer`, or `temporal`.
