# Leakage Prevention

Temporal contracts and multiple-testing discipline for time-series research.
Prevents silent look-ahead and overstated significance. Analytics remain
**measurements / evidence**, never trading signals.

## Location

- `iqrp/app/timeseries/base.py` — `TemporalMode`
- `iqrp/app/timeseries/transforms/` — leakage-safe `TimeSeriesTransformer`
- `iqrp/app/timeseries/rolling.py` — causal rolling / expanding apply
- `iqrp/app/timeseries/multiple_testing.py` — p-value adjustments

## TemporalMode contracts

| Mode | Meaning |
|------|---------|
| `POINT_IN_TIME` | Uses only the observation at t |
| `CAUSAL` | Uses ≤ t only (streaming-safe) |
| `ROLLING` | Causal fixed window ending at t |
| `EXPANDING` | Causal growing window from start → t |
| `TRAINING_ONLY` | Fit stats on train; freeze for transform |
| `FULL_SAMPLE` | May use future info — **research only** |

Every `AnalysisResult` / `DecompositionResult` / `ChangePointResult` declares
`temporal_mode`. Prefer causal modes for any feature that could enter a live
pipeline; treat `FULL_SAMPLE` as offline discovery.

## Rolling / expanding / training_only

```python
from iqrp.app.timeseries.transforms import TimeSeriesTransformer
from iqrp.app.timeseries.rolling import rolling_apply, expanding_apply
from iqrp.app.timeseries import TimeSeriesAnalyticsEngine

tf = TimeSeriesTransformer(method="zscore", window=64, temporal_mode="rolling")
z = tf.fit_transform(x)

tf2 = TimeSeriesTransformer(method="zscore", temporal_mode="training_only")
train_z = tf2.fit(train).transform(train)
test_z = tf2.transform(test)   # uses frozen train mean/std

roll_mean = rolling_apply(x, 64, float)
exp_mean = expanding_apply(x, float, min_periods=2)

eng = TimeSeriesAnalyticsEngine()  # settings.transform.temporal_mode
eng.fit(train).transform(test)
```

Hydra `transform.temporal_mode`: `rolling` | `expanding` | `training_only`.

## Multiple testing

```python
from iqrp.app.timeseries.multiple_testing import adjust_pvalues

adj = adjust_pvalues(pvalues, method="fdr_bh", alpha=0.05)
# methods: bonferroni | holm | fdr_bh | none
# returns adjusted, rejected, method, alpha, n_tests
```

`TimeSeriesAnalyticsEngine.stationarity` applies configured adjustment across
ADF/KPSS/PP/VR p-values. Never treat an unadjusted p-value from a large scan
as evidence that a feature is profitable.
