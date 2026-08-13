# Institutional Forecast Intelligence Platform

Production façade that is the **only** forecasting interface for Risk Management, Portfolio Optimization, Execution, Trading Bot, and Research.

## Location

`iqrp/app/forecasting/intelligence/`

## Responsibilities

| Capability | Module |
|------------|--------|
| Discovery | `registry.py` |
| Benchmarking | `benchmark.py` |
| Ranking / leaderboards | `ranking.py` |
| Selection | `selector.py` |
| AutoML | `automl.py`, `tuning.py`, `optimization.py` |
| Ensembles | `ensemble.py`, `stacking.py`, `blending.py`, `gating.py` |
| Routing | `routing.py` |
| Calibration | `calibration.py` |
| Uncertainty | `uncertainty.py` |
| Drift | `drift.py` |
| Retraining | `retraining.py` |
| Monitoring | `monitoring.py` |
| Deployment | `deployment.py` |
| Orchestration API | `orchestrator.py` |

## Quick start

```python
from iqrp.app.forecasting.intelligence import (
    ForecastIntelligenceEngine,
    IntelligenceSettings,
)
from iqrp.app.forecasting.intelligence.processes import simulate_market_frame, feature_names

frame = simulate_market_frame(180, kind="regime_switching", n_features=4)
feats = feature_names(4)

engine = ForecastIntelligenceEngine(
    IntelligenceSettings.from_mapping({
        "benchmark": {"method": "walk_forward", "n_splits": 2, "train_size": 60, "test_size": 20, "parallel": False},
        "ensemble": {"method": "weighted", "top_k": 1},
        "automl": {"method": "none"},
    })
)
engine.fit(frame, feature_columns=feats, candidates=["mock"])
fc = engine.forecast(frame, horizon=5)
print(engine.best_model(), engine.leaderboard()[:3])
```

## API

`fit` · `predict` · `predict_proba` · `forecast` · `forecast_interval` · `best_model` · `leaderboard` · `benchmark` · `ensemble` · `calibrate` · `monitor` · `retrain` · `save` · `load`

## Configuration

Hydra: `iqrp/configs/forecasting/intelligence/default.yaml`

```python
IntelligenceSettings.from_hydra()
IntelligenceSettings.from_mapping({"ranking": {"primary": "mae"}})
```

## Downstream contract

All consumers must call `ForecastIntelligenceEngine` — never instantiate statistical / tree / neural / transformer models directly in Risk, Portfolio, Execution, or Trading Bot paths.
