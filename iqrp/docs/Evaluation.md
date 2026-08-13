# Forecasting Evaluation

## Metrics

### Regression
MAE, MSE, RMSE, MAPE, SMAPE, R²

### Classification
Accuracy, Precision, Recall, F1, ROC AUC

### Probability
Brier Score, Log Loss, Expected Calibration Error (ECE)

### Financial
Directional Accuracy, Profit Factor, Sharpe, Sortino, Maximum Drawdown, Hit Rate

## Validation protocols

| Method | API |
|--------|-----|
| Holdout | `ForecastEvaluator.evaluate` |
| Time-series split | `time_series_splits` / `cross_validate(method="time_series_split")` |
| Walk-forward | `walk_forward_splits` / `cross_validate(method="walk_forward")` |
| Rolling | `rolling_splits` / `cross_validate(method="rolling")` |

## Benchmarking

```python
from iqrp.app.forecasting.base.evaluator import ForecastEvaluator

ev = ForecastEvaluator()
board = ev.benchmark({"mock": metrics_a, "candidate": metrics_b}, primary="rmse")
```

## Configuration

`evaluation.primary_metric` in `iqrp/configs/forecasting/default.yaml` selects the ranking key for model comparison.
