# Forecast Intelligence AutoML

Hyperparameter search over discovered forecast models.

## Methods

| Method | Implementation |
|--------|----------------|
| Grid Search | Full cartesian expansion (capped by `n_trials`) |
| Random Search | Seeded uniform draws from search space |
| Bayesian / Optuna | Optuna TPE when available; random fallback |
| Hyperband | Successive survivor elimination |
| Successive Halving | Keep top half each round |
| Population Based Training | Halving + parameter perturbation |
| Multi-objective | Optuna multi-study (`rmse` + secondary) |

## Usage

```python
from iqrp.app.forecasting.intelligence import ForecastIntelligenceEngine, IntelligenceSettings

settings = IntelligenceSettings.from_mapping({
    "automl": {"method": "random", "n_trials": 5},
    "benchmark": {"parallel": False, "n_splits": 2, "train_size": 50, "test_size": 15},
})
engine = ForecastIntelligenceEngine(settings)
engine.fit(frame, feature_columns=feats, candidates=["mock"], run_automl=True)
```

## Search spaces

`tuning.build_search_space(family=...)` returns family-specific grids for baseline, tree, neural, transformer, volatility, and statistical models.

## Facade

`optimization.run_optimization` delegates to `automl.optimize_model`.
