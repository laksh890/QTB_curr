# Feature Pipeline

## Capabilities

- Transitive dependency resolution (topological order)
- Level-wise **parallel** computation (`ThreadPoolExecutor`)
- **Caching** via content hash (memory + optional parquet directory)
- **Incremental** mode (`since=timestamp`) keeping history for rolling windows
- **Lazy** mode (resolve order only)
- Benchmarks: per-feature ms, total ms, memory estimate, cache hit rate

## Usage

```python
from iqrp.app.features import FeaturePipeline, FeatureCache
from pathlib import Path

cache = FeatureCache(directory=Path("data/feature_cache"))
pipe = FeaturePipeline(cache=cache, max_workers=4, use_cache=True)

frame, bench = pipe.compute(
    ohlcv,
    ["macd_components", "rsi", "atr", "vwap"],
    parallel=True,
)
print(bench.feature_times_ms)
print(bench.cache_hit_rate)
```

## Incremental updates

```python
frame, bench = pipe.compute(ohlcv, ["log_return", "rsi"], since=last_stored_ts)
```

The pipeline feeds `[history + new]` into rolling calcs, then filters outputs to
`open_time >= since` for storage.
