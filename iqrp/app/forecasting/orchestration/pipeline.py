"""End-to-end forecasting pipeline: preprocess → train → infer → postprocess."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.evaluator import EvaluationReport, ForecastEvaluator
from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.forecast_model import ForecastModel
from iqrp.app.forecasting.base.registry import ensure_forecast_models_loaded, get_registry
from iqrp.app.forecasting.base.trainer import ForecastTrainer, TrainResult
from iqrp.app.forecasting.config import ForecastingSettings
from iqrp.app.forecasting.postprocessing.calibration import ProbabilityCalibrator
from iqrp.app.forecasting.postprocessing.intervals import residual_intervals
from iqrp.app.forecasting.preprocessing.encoding import encode_frame_categoricals
from iqrp.app.forecasting.preprocessing.feature_selection import select_features
from iqrp.app.forecasting.preprocessing.scaling import Scaler


@dataclass(slots=True)
class PipelineResult:
    forecast: Forecast
    train: TrainResult | None = None
    evaluation: EvaluationReport | None = None
    selected_features: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ForecastingPipeline:
    """Composable training / inference pipeline over the forecasting framework."""

    def __init__(
        self,
        settings: ForecastingSettings | None = None,
        model: ForecastModel | None = None,
        model_name: str | None = None,
    ) -> None:
        self.settings = settings or ForecastingSettings.default()
        ensure_forecast_models_loaded(self.settings.discovery_modules)
        self.scaler = Scaler(kind=self.settings.preprocessing.scaler)
        self.calibrator = ProbabilityCalibrator(
            method=self.settings.postprocessing.calibration_method
        )
        self.trainer = ForecastTrainer(self.settings)
        self.evaluator = ForecastEvaluator()
        self.selected_features: list[str] = []
        self._encoders: dict[str, Any] = {}
        if model is not None:
            self.model = model
        elif model_name is not None:
            self.model = get_registry().create(model_name, settings=self.settings)
        else:
            names = get_registry().list_names()
            if not names:
                from iqrp.app.core.exceptions import ConfigurationError

                raise ConfigurationError(
                    "No forecasting models registered",
                    code="FC_NO_MODELS",
                )
            self.model = get_registry().create(names[0], settings=self.settings)

    def preprocess(self, frame: pl.DataFrame, *, fit: bool = False) -> pl.DataFrame:
        s = self.settings
        out = frame
        if s.preprocessing.encode_categoricals:
            out, enc = encode_frame_categoricals(out)
            if fit:
                self._encoders = enc
        cols, tgt = self.trainer.resolve_columns(
            out, list(s.columns.feature_columns) if s.columns.feature_columns else None, s.columns.target
        )
        if not cols:
            return out
        x = out.select(cols).to_numpy().astype(np.float64)
        y = out[tgt].to_numpy() if tgt and tgt in out.columns else None
        if fit:
            idx = select_features(
                x,
                y,
                method=s.preprocessing.feature_selection,
                max_features=s.preprocessing.max_features,
                variance_threshold=s.preprocessing.variance_threshold,
                correlation_threshold=s.preprocessing.correlation_threshold,
            )
            self.selected_features = [cols[i] for i in idx]
            x_sel = x[:, idx]
            self.scaler.fit(x_sel)
        else:
            if not self.selected_features:
                self.selected_features = cols
            idx = [cols.index(c) for c in self.selected_features if c in cols]
            x_sel = x[:, idx] if idx else x
        x_t = self.scaler.transform(x_sel)
        # write scaled columns back under selected names
        for j, name in enumerate(self.selected_features):
            if j < x_t.shape[1]:
                out = out.with_columns(pl.Series(name=name, values=x_t[:, j]))
        return out

    def fit(
        self,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> TrainResult:
        prepared = self.preprocess(frame, fit=True)
        cols = feature_columns or self.selected_features
        tgt = target_column or self.settings.columns.target
        regime = regime_column or self.settings.columns.regime_column
        result = self.trainer.fit(
            self.model,
            prepared,
            feature_columns=cols,
            target_column=tgt,
            regime_column=regime if regime and regime in prepared.columns else None,
            validate=True,
        )
        return result

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        prepared = self.preprocess(frame, fit=False)
        return self.model.predict(prepared, self.selected_features or None)

    def forecast(self, frame: pl.DataFrame, *, horizon: int | None = None) -> Forecast:
        prepared = self.preprocess(frame, fit=False)
        h = horizon or self.settings.inference.default_horizon
        fc = self.model.forecast(
            prepared, horizon=h, feature_columns=self.selected_features or None
        )
        if fc.intervals is None and self.settings.postprocessing.interval_level:
            fc.intervals = residual_intervals(
                fc.path(), level=self.settings.postprocessing.interval_level
            )
        fc.strategy = self.settings.inference.strategy
        fc.features_used = tuple(self.selected_features)
        return fc

    def run(
        self,
        train_frame: pl.DataFrame,
        infer_frame: pl.DataFrame | None = None,
        *,
        horizon: int | None = None,
    ) -> PipelineResult:
        train = self.fit(train_frame)
        frame = infer_frame if infer_frame is not None else train_frame
        fc = self.forecast(frame, horizon=horizon)
        evaluation = None
        tgt = self.settings.columns.target
        if tgt and tgt in frame.columns:
            evaluation = self.evaluator.evaluate(
                frame[tgt].to_numpy(), self.predict(frame), task=self.model.meta.task
            )
        return PipelineResult(
            forecast=fc,
            train=train,
            evaluation=evaluation,
            selected_features=list(self.selected_features),
        )
