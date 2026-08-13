# Dynamic Ensembling

Combine ranked forecast models under the intelligence engine.

## Methods

| Method | Module |
|--------|--------|
| Weighted average | `ensemble.weighted_average` |
| Median | `ensemble.median_ensemble` |
| Bayesian model averaging | `ensemble.bayesian_model_averaging` |
| Voting | `ensemble.voting_ensemble` |
| Stacking | `stacking.stack_predictions` |
| Blending | `blending.blend_predictions` |
| Mixture of Experts | `gating.moe_combine` |
| Dynamic ensemble selection | `ensemble.dynamic_ensemble_selection` |

## Routing

`routing.route_model` switches experts by asset, regime, volatility, liquidity (spread), timeframe, and confidence.

## Usage

```python
settings = IntelligenceSettings.from_mapping({"ensemble": {"method": "bma", "top_k": 3}})
engine = ForecastIntelligenceEngine(settings)
engine.fit(frame, feature_columns=feats, candidates=["mock"])
path = engine.ensemble(frame)
fc = engine.forecast(frame, horizon=5)  # uses ensemble when multiple members fit
```
