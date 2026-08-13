# Stationarity

Unit-root and random-walk tests for analytical evidence of persistence.
Outputs are hypothesis-test measurements — **not trading signals**.

## Location

`iqrp/app/timeseries/stationarity/`

- `adf.py` — Augmented Dickey–Fuller
- `kpss.py` — KPSS
- `phillips_perron.py` — Phillips–Perron
- `variance_ratio.py` — Lo–MacKinlay variance ratio

## Tests

| Test | H₀ | Rejection suggests |
|------|----|--------------------|
| ADF | unit root (non-stationary) | stationary |
| KPSS | stationary | unit root |
| Phillips–Perron | unit root | stationary |
| Variance ratio | random walk (VR = 1) | VR ≠ 1 (mean reversion or momentum) |

```python
from iqrp.app.timeseries.stationarity.adf import adf
from iqrp.app.timeseries.stationarity.kpss import kpss
from iqrp.app.timeseries.stationarity.phillips_perron import phillips_perron
from iqrp.app.timeseries.stationarity.variance_ratio import variance_ratio

a = adf(x, max_lag=None, alpha=0.05)       # Schwert lag if max_lag is None
k = kpss(x, regression="c", alpha=0.05)   # "ct" for constant+trend
p = phillips_perron(x, alpha=0.05)
v = variance_ratio(x, lags=2, alpha=0.05) # value = VR; metadata notes VR<1 / VR>1
```

## Engine battery

```python
from iqrp.app.timeseries import TimeSeriesAnalyticsEngine

report = TimeSeriesAnalyticsEngine().stationarity(x)
# report["tests"]["adf"|"kpss"|"phillips_perron"|"variance_ratio"]
# report["multiple_testing"] — FDR/Holm/Bonferroni on collected p-values
```

## Interpretation

- Prefer **joint reading**: ADF rejects + KPSS fails to reject → evidence of
  stationarity; the reverse → evidence of a unit root.
- Conflicting ADF/KPSS outcomes are common near the boundary — treat as
  inconclusive, not a feature.
- VR < 1 → mean-reverting increments; VR > 1 → trending; neither is a signal.
- All tests are `TemporalMode.FULL_SAMPLE`. Use adjusted p-values from
  `multiple_testing` before claiming significance across a research battery.
