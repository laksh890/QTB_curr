# Forecast Intelligence Benchmarking

Time-series-safe evaluation of every candidate model.

## Protocols

| Method | Description |
|--------|-------------|
| `walk_forward` | Expanding / stepped train→test windows |
| `rolling` | Fixed-size rolling train window |
| `time_series_split` | Expanding fraction splits |
| `nested_cv` | Purged folds (outer selection) |
| `purged_kfold` | Purge window around test fold |
| `embargo` | Purge + embargo gap after test |

## Metrics

MAE, RMSE, MAPE, SMAPE, directional accuracy, Sharpe, Sortino, profit factor, max drawdown, log loss, Brier, calibration error, prediction stability, latency, memory, inference cost.

## Parallelism

`benchmark.parallel=true` runs candidates via `ThreadPoolExecutor` (`max_workers`).

## API

```python
engine.benchmark(frame, feature_columns=feats, candidates=["mock"])
engine.leaderboard()
```
