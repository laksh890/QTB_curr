"""Prediction / forecast service for fitted regime models."""

from __future__ import annotations

import numpy as np
import polars as pl

from iqrp.app.regimes.base.forecast import RegimeForecast
from iqrp.app.regimes.base.regime_model import RegimeModel
from iqrp.app.regimes.config import RegimeSettings


class RegimePredictor:
    def __init__(self, settings: RegimeSettings | None = None) -> None:
        self.settings = settings or RegimeSettings.default()

    def predict(
        self,
        model: RegimeModel,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> np.ndarray:
        return model.predict(frame, feature_columns)

    def predict_proba(
        self,
        model: RegimeModel,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> np.ndarray:
        return model.predict_proba(frame, feature_columns)

    def forecast(self, model: RegimeModel, frame: pl.DataFrame, steps: int = 1) -> RegimeForecast:
        return model.forecast(frame, steps=steps)

    def transition_matrix(self, model: RegimeModel) -> np.ndarray:
        return model.transition_matrix()
