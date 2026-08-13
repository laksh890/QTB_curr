# Anomaly Detection

Flag unusual points or subsequences for analytical review.
Anomaly indices are **evidence of irregularity**, not buy/sell signals.

## Location

`iqrp/app/timeseries/anomaly/`

- `statistical.py` — z-score thresholds (full-sample or rolling)
- `robust.py` — median / MAD robust z-score
- `isolation_forest.py` — Isolation Forest (sklearn or NumPy fallback)
- `matrix_profile.py` — high matrix-profile discords as anomalies

## Methods

```python
from iqrp.app.timeseries.anomaly.statistical import zscore_anomalies
from iqrp.app.timeseries.anomaly.robust import robust_zscore_anomalies
from iqrp.app.timeseries.anomaly.isolation_forest import isolation_forest_anomalies
from iqrp.app.timeseries.anomaly.matrix_profile import matrix_profile_anomalies

z = zscore_anomalies(x, threshold=3.0, window=None)      # FULL_SAMPLE if window None
r = robust_zscore_anomalies(x, threshold=3.0, window=None)
iso = isolation_forest_anomalies(x, contamination=0.05, window=1)
mp = matrix_profile_anomalies(x, window=32)
```

Engine (`anomaly.method`: `statistical|robust|isolation_forest|matrix_profile`):

```python
from iqrp.app.timeseries import TimeSeriesAnalyticsEngine

an = TimeSeriesAnalyticsEngine().anomalies(x, method="robust")
# an.value — anomaly indices (or structured payload); check metadata
```

Hydra: `anomaly.z_threshold`, `anomaly.contamination`; motif window reused for
matrix-profile mode.

## Temporal modes

| Method | Mode |
|--------|------|
| z / robust with `window=None` | `FULL_SAMPLE` |
| z / robust with rolling `window` | `ROLLING` |
| Isolation Forest / matrix profile | `FULL_SAMPLE` |

Prefer rolling robust scores for any causal monitoring. Full-sample Isolation
Forest and matrix profile are retrospective discovery tools only.
