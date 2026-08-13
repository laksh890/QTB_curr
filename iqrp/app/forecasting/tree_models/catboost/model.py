"""CatBoost forecasting model."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.tree_models.base.backends import BackendName
from iqrp.app.forecasting.tree_models.base.tree_model import TreeForecastModel


@register_forecast_model
class CatBoostForecastModel(TreeForecastModel):
    backend: BackendName = "catboost"
    meta = ForecastModelMeta(
        name="catboost",
        version="1.0.0",
        description="CatBoost ordered boosting",
        algorithm_family="tree",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_proba=True,
        supports_intervals=True,
        supports_quantiles=True,
    )

    def __init__(self, settings: Any | None = None, **params: Any) -> None:
        super().__init__(settings=settings, **params)

    def _backend_name(self) -> BackendName:
        return "catboost"
