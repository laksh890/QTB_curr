# Nonlinear Analysis

Nonlinear / complexity descriptors for series structure.

> **Disclaimer:** These are **statistical descriptors only** — not guaranteed
> predictive signals and **not trading signals**. Treat Hurst, fractal
> dimension, and entropies as summary measurements for research batteries.

## Location

`iqrp/app/timeseries/nonlinear/`

- `hurst.py` — R/S Hurst exponent
- `fractal_dimension.py` — Higuchi FD
- `entropy.py` — Shannon entropy (histogram)
- `sample_entropy.py` — Sample Entropy
- `approximate_entropy.py` — Approximate Entropy
- `permutation_entropy.py` — Permutation Entropy

## API

```python
from iqrp.app.timeseries.nonlinear.hurst import hurst_exponent
from iqrp.app.timeseries.nonlinear.fractal_dimension import higuchi_fd
from iqrp.app.timeseries.nonlinear.entropy import shannon_entropy
from iqrp.app.timeseries.nonlinear.sample_entropy import sample_entropy
from iqrp.app.timeseries.nonlinear.approximate_entropy import approximate_entropy
from iqrp.app.timeseries.nonlinear.permutation_entropy import permutation_entropy

H = hurst_exponent(x)          # H≈0.5 RW; H>0.5 persistence; H<0.5 anti-persistence
fd = higuchi_fd(x, k_max=10)   # FD≈1 smooth; higher → more complex path
sh = shannon_entropy(x)
se = sample_entropy(x)
ae = approximate_entropy(x)
pe = permutation_entropy(x)
```

Engine:

```python
from iqrp.app.timeseries import TimeSeriesAnalyticsEngine

eng = TimeSeriesAnalyticsEngine()
H = eng.hurst(x)
ents = eng.entropy(x)
# ents includes shannon/sample/approximate/permutation + disclaimer string
```

## Interpretation (descriptive only)

| Metric | Typical reading |
|--------|-----------------|
| Hurst H | Memory / roughness of increments |
| Higuchi FD | Geometric complexity of the path |
| Entropies | Irregularity / unpredictability of patterns |

All methods use `TemporalMode.FULL_SAMPLE`. Never promote raw descriptor
values into live alphas without proper causal validation and multiple-testing
control.
