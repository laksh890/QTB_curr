"""Institutional Tree-Based Machine Learning Forecasting Engine.

Regression, classification, quantile and probability forecasting via
XGBoost, LightGBM, CatBoost and sklearn ensembles — all inheriting from
the Forecasting Framework.
"""

from iqrp.app.forecasting.tree_models.config import TreeSettings
from iqrp.app.forecasting.tree_models.registry import (
    create_tree_model,
    ensure_tree_models_loaded,
    list_tree_models,
)
from iqrp.app.forecasting.tree_models.trainer import TreeTrainer, TreeTrainResult

ensure_tree_models_loaded()

__all__ = [
    "TreeSettings",
    "TreeTrainResult",
    "TreeTrainer",
    "create_tree_model",
    "ensure_tree_models_loaded",
    "list_tree_models",
]
