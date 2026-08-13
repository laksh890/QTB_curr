# Consensus

Disagreement and consensus analysis across ensemble members.

## Metrics

| Metric | Meaning |
|--------|---------|
| Pairwise JS disagreement | Mean Jensen–Shannon divergence between members |
| Consensus score | `1 - disagreement / ln(2)` |
| Hard agreement | Fraction agreeing with majority label |
| Prediction diversity | Unique hard-path fraction |
| Mean entropy | Average member predictive entropy |

## API

```python
model.consensus(frame)
model.confidence(frame)
```

## Location

`disagreement.py`, `confidence.py`
