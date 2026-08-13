# Feature Store

## Layout

```text
{root}/
  exchange={ex}/
    symbol={sym}/
      timeframe={tf}/
        feature_group={group}/
          year=YYYY/
            month=MM/
              features.parquet
```

Compression: ZSTD.

## Operations

```python
from pathlib import Path
from iqrp.app.features import FeatureStore, FeatureQueryService

store = FeatureStore(Path("data/features"))
store.write(frame, exchange="binance", symbol="BTCUSDT", timeframe="1m", feature_group="trend")
store.update_incremental(new_frame, exchange="binance", symbol="BTCUSDT", timeframe="1m", feature_group="trend")

svc = FeatureQueryService(store=store)
svc.get_feature("log_return", exchange="binance", symbol="BTCUSDT", timeframe="1m")
svc.get_features(["rsi", "atr"], exchange="binance", symbol="BTCUSDT", timeframe="1m")
svc.list_features(category="momentum")
svc.describe_feature("rsi")
svc.feature_dependencies("macd_components")
```

`compute_and_store` runs the pipeline then writes (full or incremental).
