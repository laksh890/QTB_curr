# Feature Research Validation Guide

## Questions this engine answers

| Question | Evidence |
|----------|----------|
| Does Feature A predict returns? | Walk-forward IC / Rank IC / R² vs `future_return` |
| Does Feature B predict volatility? | Same metrics vs `future_volatility` |
| Is Feature C redundant? | VIF, near-duplicate corr, rolling-window families |
| Is Feature D useful across regimes? | Stability rolling IC + drift concept IC ratio |
| Is Feature E stable over time? | Rolling mean/variance stability, decay, parameter drift |

## Scoring (0–100)

Configurable weights in `configs/research/default.yaml` → `scoring.*`:

- Predictive power
- Stability
- Redundancy penalty (subtracted)
- Computational cost (subtracted)
- Interpretability
- Consistency across assets / timeframes (caller-supplied maps)

Decisions:

- `accept` if score ≥ `accept_score_threshold` and not suggested for removal
- `reject` if score ≤ `reject_score_threshold` or constant/empty
- `weak` otherwise

## API

```python
from iqrp.app.features.research import FeatureResearchValidator, ResearchSettings

settings = ResearchSettings.from_hydra(overrides=["scoring.accept_score_threshold=65"])
result = FeatureResearchValidator(settings).validate(frame)

result.rankings["most_predictive_features"]
result.accepted()
result.report_paths["markdown"]
```

## Time-series integrity

Never shuffle. Evaluation modes: `walk_forward`, `expanding`, `rolling`, `blocked`.

## Optional dependencies

- `minepy` → MIC
- `shap` + `scikit-learn` → Tree SHAP (else linear SHAP-lite)
