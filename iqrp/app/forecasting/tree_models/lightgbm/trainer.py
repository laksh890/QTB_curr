"""LightGBM-specific training helpers."""

from __future__ import annotations

from typing import Any

import polars as pl

from iqrp.app.forecasting.tree_models.config import TreeSettings
from iqrp.app.forecasting.tree_models.lightgbm.model import LightGBMForecastModel


class LightGBMTrainer:
    def __init__(self, settings: TreeSettings | None = None) -> None:
        self.settings = settings or TreeSettings.default()

    def fit(
        self,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        **params: Any,
    ) -> LightGBMForecastModel:
        model = LightGBMForecastModel(settings=self.settings, **params)
        model.fit(frame, feature_columns, target_column=target_column)
        return model
