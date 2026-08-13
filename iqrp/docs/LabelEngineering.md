# Label Engineering Platform

Production-grade prediction-target generation for IQRP.

**Scope:** label generation only. No machine learning. No trading strategies.

## Location

`iqrp/app/labels/`

## Configuration

Hydra defaults: `iqrp/configs/labels/default.yaml`

```python
from iqrp.app.labels import LabelSettings, LabelPipeline

settings = LabelSettings.from_hydra(overrides=["defaults.horizon=24"])
```

## Workflow

```python
from iqrp.app.labels import LabelPipeline, LabelQueryService, list_labels

print(list_labels(category="regression"))
out, bench = LabelPipeline().compute(ohlcv, ["future_return", "binary_up", "triple_barrier"])

svc = LabelQueryService()
svc.compute_and_store(ohlcv, ["future_return", "triple_barrier"],
                      exchange="binance", symbol="BTCUSDT", timeframe="1h")
```

## Categories

| Category | Examples |
|----------|----------|
| regression | future return/log-return/vol/ATR/drawdown/MFE/MAE/VWAP deviation/spread/liquidity |
| classification | binary up/down, return/vol/trend buckets, regime class, market stress |
| survival | time-to upper/lower barrier |
| volatility | realized, Parkinson, Garman-Klass, Yang-Zhang, EWMA |
| regime | bull/bear/sideways, vol/liquidity/trend regimes |
| barrier | full Triple Barrier Method |
| meta | meta-label, probability label, trade filter |
| custom | user-registered horizons / probability targets |

## Query API

- `get_label` / `get_labels`
- `describe_label` / `list_labels`

## Storage

Versioned Parquet under `store_dir`, partitioned by exchange/symbol/timeframe/label/version/year/month, with incremental updates.

## Validation & quality

`LabelValidator` detects look-ahead anomalies, leakage heuristics, missing/duplicate/imbalanced/degenerate labels and emits quality metrics (distribution, entropy, coverage, horizon).

## Related docs

- [TripleBarrier.md](TripleBarrier.md)
- [MetaLabeling.md](MetaLabeling.md)
