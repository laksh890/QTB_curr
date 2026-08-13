# Institutional Tree-Based Forecasting Engine

Production tree / boosting forecasts for IQRP. All models inherit from the Forecasting Framework (`ForecastModel` → `TreeForecastModel`).

## Location

`iqrp/app/forecasting/tree_models/`

## Models

| Name | Backend |
|------|---------|
| `xgboost` | XGBoost |
| `lightgbm` | LightGBM |
| `catboost` | CatBoost |
| `hist_gradient_boosting` | sklearn HistGBM |
| `random_forest` | sklearn RF |
| `extra_trees` | sklearn ExtraTrees |

Native numpy fallbacks activate automatically if a library is unavailable.

## Tasks

Regression · Binary / multiclass classification · Quantile regression · Probability estimation

## Quick start

```python
from iqrp.app.forecasting.tree_models import (
    TreeSettings, TreeTrainer, create_tree_model,
)

model = create_tree_model("xgboost")
model.fit(frame, feature_columns=["f0", "f1", "f2"], target_column="target")
pred = model.predict(frame)
fc = model.forecast(frame, horizon=5)
imp = model.feature_importance(kind="gain")
sv = model.shap_values(frame)
```

## API

`fit` · `partial_fit` · `predict` · `predict_proba` · `forecast` · `forecast_interval` · `evaluate` · `explain` · `feature_importance` · `shap_values` · `diagnostics` · `cross_validate` · `save` / `load`

## Configuration

Hydra: `iqrp/configs/forecasting/tree_models/default.yaml`

```python
TreeSettings.from_hydra(overrides=["task.type=binary", "calibration.enabled=true", "calibration.method=platt"])
```

## Integrations

- Validated features / Feature Store columns as `feature_columns`
- Regime Intelligence via `regime` column (`feature` / `separate` / `weighted` / `routing`)
- Volatility forecasts as input features (e.g. `vol_forecast`)
- Label Platform targets via `target_column`
- Serialization via Forecasting Framework `ForecastSerializer`
