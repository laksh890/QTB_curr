# Dependence Analysis

Bivariate / pairwise dependence measurements for co-movement research.
Results are **statistical evidence of association**, not trading signals
(including “pairs” narratives).

## Location

`iqrp/app/timeseries/dependence/`

- `cointegration.py` — Engle–Granger; Johansen trace (two series)
- `granger.py` — Granger causality F-test
- `mutual_information.py` — histogram MI
- `distance_correlation.py` — distance correlation
- `tail_dependence.py` — empirical upper/lower tail dependence

## API

```python
from iqrp.app.timeseries.dependence.cointegration import engle_granger, johansen_trace
from iqrp.app.timeseries.dependence.granger import granger_causality
from iqrp.app.timeseries.dependence.mutual_information import mutual_information
from iqrp.app.timeseries.dependence.distance_correlation import distance_correlation
from iqrp.app.timeseries.dependence.tail_dependence import empirical_tail_dependence

eg = engle_granger(x, y)
jo = johansen_trace(x, y)
gc = granger_causality(x, y, max_lag=2)   # does x Granger-cause y?
mi = mutual_information(x, y)
dc = distance_correlation(x, y)
td = empirical_tail_dependence(x, y)
```

Engine:

```python
from iqrp.app.timeseries import TimeSeriesAnalyticsEngine

eng = TimeSeriesAnalyticsEngine()
coint = eng.cointegration(x, y, method="engle_granger")  # or "johansen"
dep = eng.dependence(x, y)
# keys: granger, mutual_information, distance_correlation,
#       tail_dependence, cointegration
```

## Notes

- **Cointegration** — shared stochastic trend evidence; not a mean-reversion trade.
- **Granger** — predictive linear lag association under a VAR-style F-test;
  not structural causality and not a signal.
- **MI / distance corr** — capture nonlinear association; scale-free summaries.
- **Tail dependence** — co-crash / co-rally frequency in empirical tails.

All methods are `TemporalMode.FULL_SAMPLE`. Apply multiple-testing adjustments
when screening many pairs.
