# Decomposition

Seasonal-trend-residual decomposition for analytical discovery.
**Not forecasting** — components are measurements of structure, not signals.

## Location

`iqrp/app/timeseries/decomposition/`

- `classical.py` — centered MA + seasonal means
- `stl.py` — LOESS-lite STL (additive)
- `mstl.py` — multiple seasonal periods via iterative STL
- `trend.py` / `seasonal.py` — helpers and strength metrics

## Methods

| Method | Model | Notes |
|--------|-------|-------|
| `classical` | additive / multiplicative | Full-sample centered MA trend |
| `stl` | additive only | Iterative LOESS seasonal + trend; optional robust weights |
| `mstl` | additive | Sum of seasonal components over `periods` |

```python
from iqrp.app.timeseries.decomposition.classical import classical_decompose
from iqrp.app.timeseries.decomposition.stl import stl_decompose
from iqrp.app.timeseries.decomposition.mstl import mstl_decompose

classic = classical_decompose(x, period=24, model="additive")
# model="multiplicative": Y = T × S × R  (else Y = T + S + R)

stl = stl_decompose(x, period=24, robust=True, n_iter=2)
mstl = mstl_decompose(x, periods=(24, 168), robust=False)
```

Via engine (defaults from Hydra `decomposition.*`):

```python
from iqrp.app.timeseries import TimeSeriesAnalyticsEngine

eng = TimeSeriesAnalyticsEngine()
res = eng.decompose(x, method="stl", period=24)
# res.trend, res.seasonal, res.residual, res.observed
```

## Additive vs multiplicative

- **Additive** — constant seasonal amplitude; residual = observed − trend − seasonal.
- **Multiplicative** — seasonal scales with level; only supported by classical.
  STL/MSTL always return `model="additive"`.

## Temporal contract

All decompositions use `TemporalMode.FULL_SAMPLE` (centered / bidirectional
smoothers). Do not feed components into live features without a causal redesign.

Result type: `DecompositionResult` (`to_dict()` for serialization).
