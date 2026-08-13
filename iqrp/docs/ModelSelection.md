# Forecast Intelligence Model Selection

Automatic selection of model, horizon, feature set, regime specialists, and volatility specialists.

> Note: Volatility-family AIC/BIC selection remains documented under Volatility Forecasting. This page covers the **intelligence-layer** selector that ranks across all discovered engines.

## What is selected

| Target | Logic |
|--------|-------|
| Best model | Composite ranking over benchmark folds |
| Best features | Greedy drop-one RMSE improvement |
| Best horizon | Forecast-path stability search |
| Regime models | Per-regime re-benchmark when `regime` column present |
| Volatility model | Top ranked model with `family == volatility` |
| Ensemble method | From `ensemble.method` settings |

## Usage

```python
engine.fit(frame, feature_columns=feats, candidates=["mock"], run_selection=True)
print(engine.best_model())
print(engine._selection.to_dict() if engine._selection else {})
```

## Ranking

Lower composite score wins. Higher-is-better metrics (Sharpe, directional accuracy, …) are negated inside `ranking.composite_score`.
