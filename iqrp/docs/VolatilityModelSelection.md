# Volatility Model Selection

## Automatic comparison

```python
from iqrp.app.forecasting.volatility import VolatilityTrainer

trainer = VolatilityTrainer()
model, result = trainer.auto_select(frame, candidates=["ewma", "arch", "garch", "gjr_garch", "egarch"])
print(result.selection.best, result.selection.leaderboard)
```

## Criteria

| Criterion | Direction | Source |
|-----------|-----------|--------|
| `aic` | minimize | in-sample MLE |
| `bic` | minimize | in-sample MLE |
| `loglik` | maximize | in-sample MLE |
| `qlike` | minimize | variance forecast loss |

Configured via `selection_criterion` in Hydra / `VolatilitySettings`.

## Rolling validation

`base/selection.py::rolling_vol_validation` walks a train window, produces h-step variance forecasts, and reports RMSE / QLIKE.

## Parallel fitting

`VolatilityTrainer.compare` and `select_volatility_models` use a thread pool for concurrent model fits on large datasets.

Cross-engine selection across statistical / tree / neural / transformers is handled by Forecast Intelligence — see `ModelSelection.md` and `ForecastIntelligence.md`.
