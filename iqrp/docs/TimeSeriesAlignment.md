# Time-Series Alignment

Shape-based similarity and discriminative subsequence discovery.
Distances and shapelets are **geometric measurements**, not trading signals.

## Location

`iqrp/app/timeseries/alignment/`

- `dtw.py` — Dynamic Time Warping distance (+ optional path)
- `soft_dtw.py` — Soft-DTW (differentiable / smoothed alignment)
- `shapelets.py` — shapelet discovery (supervised or unsupervised)

## DTW and Soft-DTW

```python
from iqrp.app.timeseries.alignment.dtw import dtw_distance, dtw_path
from iqrp.app.timeseries.alignment.soft_dtw import soft_dtw

d = dtw_distance(x, y)
path = dtw_path(x, y)          # alignment path in metadata / value
s = soft_dtw(x, y)             # smoothed DTW cost
```

Engine:

```python
from iqrp.app.timeseries import TimeSeriesAnalyticsEngine

eng = TimeSeriesAnalyticsEngine()
hard = eng.dtw(x, y, soft=False)
soft = eng.dtw(x, y, soft=True)
```

## Shapelets

```python
from iqrp.app.timeseries.alignment.shapelets import discover_shapelets

sh = discover_shapelets(
    x,
    labels=None,                 # optional class labels → info-gain scoring
    lengths=(8, 16, 32),
    top_k=3,
    n_candidates=50,
)
# without labels: high local-vs-global variance contrast
```

## Temporal contract

Alignment and shapelet discovery are `TemporalMode.FULL_SAMPLE`. Pairwise
distances summarize historical shape similarity; they do not imply future
co-movement or tradeable edges.
