# Feature Engineering

## Role

`iqrp.app.features` is the **central feature store** for IQRP. Downstream models
(Markov, HMM, Bayesian, gradient boosting, transformers, RL) must consume
features exclusively through this package.

This module generates features only — no signals, strategies, portfolio logic,
or model training.

## Layout

```text
app/features/
  base/           Feature, Registry, Pipeline, Cache
  trend/ …        Category implementations
  transforms/     Lag, diff, normalize, Box-Cox, …
  validation/     NaN/Inf/correlation checks
  metadata/       Catalog helpers
  store/          Parquet feature store
  query.py        get_feature(s), list/describe/deps
```

## Quick start

```python
import polars as pl
from iqrp.app.features import FeaturePipeline, FeatureQueryService, list_features

print(list_features(category="trend"))

pipe = FeaturePipeline()
enriched, bench = pipe.compute(ohlcv_frame, ["log_return", "rsi", "atr"])
print(bench.to_dict())

svc = FeatureQueryService(store_root=Path("data/features"))
svc.compute_and_store(
    ohlcv_frame,
    ["log_return", "rsi"],
    exchange="binance",
    symbol="BTCUSDT",
    timeframe="1m",
    feature_group="momentum",
)
```

## Contracts

- Primary structure: **Polars DataFrames only**
- Every feature declares metadata (name, version, deps, columns, category)
- Pipeline resolves dependencies, supports parallel batches, caching, incremental windows
- Persistence: Parquet partitions by exchange/symbol/timeframe/feature_group/year/month
