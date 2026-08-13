# Persistence Analysis

State duration and occupancy statistics for Markov regimes.

## Location

`iqrp/app/regimes/markov/persistence.py`

## Metrics

| Metric | Definition |
|--------|------------|
| Average state duration | Empirical mean run length per state |
| Expected duration | Geometric `1 / (1 - P_ii)` |
| Persistence score | `P_ii` |
| Transition frequency | Count matrix + switch rate |
| State occupancy | Empirical frequencies |

```python
report = model.persistence_report(states)
print(report["expected_duration"], report["transition_frequency"]["switch_rate"])
```

Used by diagnostics and SVG persistence histograms.
