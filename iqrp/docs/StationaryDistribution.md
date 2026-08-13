# Stationary Distribution

Steady-state analysis for discrete Markov chains.

## Location

`iqrp/app/regimes/markov/stationary.py`

## Outputs

| Quantity | Meaning |
|----------|---------|
| Stationary distribution `π` | Solve `π P = π`, `π 1 = 1` (math-engine) |
| Steady-state probabilities | Alias of `π` |
| Irreducibility | Strong connectivity of `P` support |
| Aperiodicity / period | Self-loops or gcd of return times |
| Ergodicity | Irreducible + aperiodic |
| Mixing time | Spectral-gap bound from math-engine |
| Spectral gap | `1 - |λ₂|` |

```python
from iqrp.app.regimes.markov import MarkovChainModel

model.fit(states)
report = model.stationary_analysis()
print(report["stationary_distribution"], report["is_ergodic"], report["mixing_time"])
```
