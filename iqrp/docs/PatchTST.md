# PatchTST

Channel-independent patching of lookback windows into tokens, then a Transformer encoder over patches.

```python
settings = TransformerSettings.from_mapping({"architecture": {"patch_len": 16, "stride": 8}})
model = create_transformer_model("patchtst", settings=settings)
```
