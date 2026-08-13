# XGBoost in IQRP

Model name: `xgboost`  
Class: `iqrp.app.forecasting.tree_models.xgboost.model.XGBoostForecastModel`  
Trainer: `iqrp.app.forecasting.tree_models.xgboost.trainer.XGBoostTrainer`

## Defaults

Configured via `TreeSettings.hyperparameters` — `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`.

GPU: set `hyperparameters.device=cuda` (uses `device=cuda`, `tree_method=hist`).

## Tasks

- Regression: `reg:squarederror`
- Classification: `binary:logistic`
- Quantile: `reg:quantileerror` with `task.quantile_alphas`

## Early stopping

When `optimization.early_stopping=true`, fit uses an internal holdout `eval_set` when the backend supports it.
