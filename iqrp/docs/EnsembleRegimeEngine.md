# Ensemble Regime Engine

Institutional Market Regime Intelligence Engine — the **only** regime interface
for downstream forecasting, risk, portfolio optimization, and execution.

## Location

`iqrp/app/regimes/ensemble/`

## Discovery

Members are discovered from the global regime registry after importing modules
listed in Hydra `discovery_modules`. No hard-coded model imports.

```python
from iqrp.app.regimes.ensemble import EnsembleRegimeModel, EnsembleSettings

model = EnsembleRegimeModel(settings=EnsembleSettings.from_hydra(
    overrides=["discovery_modules=[iqrp.app.regimes.models.mock]"]
))
model.fit(frame)
proba = model.predict_proba(frame)
fc = model.forecast(frame, steps=5)
print(model.weights(), model.confidence(), model.leaderboard())
```

## Canonical regimes

Default: `bull`, `bear`, `sideways`, `high_volatility`, `low_volatility`,
`liquidity_stress`. Each member maps into this space via name aliases / soft maps.

## Integration

Registered as `"ensemble"` on both Regime and State Space registries.
Hydra: `iqrp/configs/ensemble/default.yaml`.
