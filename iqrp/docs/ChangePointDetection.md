# Change-Point Detection

Detect structural shifts in mean (and related online variants).
Indices and scores are **measurements of instability**, not entry/exit signals.

## Location

`iqrp/app/timeseries/change_points/`

- `cusum.py` — offline two-sided CUSUM (FULL_SAMPLE)
- `binary_segmentation.py` — BinSeg with L2 mean cost
- `pelt.py` — PELT (pruned exact linear time)
- `bayesian.py` — Adams–MacKay BOCPD (CAUSAL)
- `online.py` — streaming Page–Hinkley / online CUSUM (CAUSAL)

## Methods

```python
from iqrp.app.timeseries.change_points.cusum import cusum_detect
from iqrp.app.timeseries.change_points.binary_segmentation import binseg_detect
from iqrp.app.timeseries.change_points.pelt import pelt_detect
from iqrp.app.timeseries.change_points.bayesian import bayesian_online_changepoint
from iqrp.app.timeseries.change_points.online import online_cusum

cp = cusum_detect(x)                          # bridge-adjusted CUSUM peaks
bs = binseg_detect(x, max_cps=5, min_size=10)
pe = pelt_detect(x, penalty=3.0, min_size=10)
bo = bayesian_online_changepoint(x, hazard=1/200, threshold=0.5)
on = online_cusum(x, threshold=5.0, drift=0.5, warmup=20)
```

Via engine (`change_points.method` in Hydra: `cusum|binseg|pelt|bayesian|online`):

```python
from iqrp.app.timeseries import TimeSeriesAnalyticsEngine

res = TimeSeriesAnalyticsEngine().change_points(x, method="pelt")
# res.indices, res.scores, res.kind, res.temporal_mode
```

## Temporal modes

| Method | Mode | Notes |
|--------|------|-------|
| CUSUM / BinSeg / PELT | `FULL_SAMPLE` | Use full-series mean/cost — research only |
| Bayesian BOCPD | `CAUSAL` | Run-length posterior uses past only |
| Online CUSUM | `CAUSAL` | Pass `state` for streaming batches |

## Result

`ChangePointResult`: `indices`, optional `scores`, `kind="mean"`, parameters,
metadata. Prefer causal methods for any live monitoring; offline methods for
retrospective structural analysis only.
