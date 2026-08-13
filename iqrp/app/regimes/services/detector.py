"""High-level regime detection orchestration service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from iqrp.app.regimes.base.evaluator import EvaluationReport
from iqrp.app.regimes.base.regime import RegimeResult
from iqrp.app.regimes.base.regime_model import RegimeModel
from iqrp.app.regimes.base.registry import ensure_regime_models_loaded, get_registry
from iqrp.app.regimes.config import RegimeSettings
from iqrp.app.regimes.services.predictor import RegimePredictor
from iqrp.app.regimes.services.serializer import RegimeSerializer
from iqrp.app.regimes.services.trainer import RegimeTrainer
from iqrp.app.regimes.storage.regime_store import RegimeStore
from iqrp.app.regimes.visualization.persistence import plot_persistence
from iqrp.app.regimes.visualization.probabilities import plot_probabilities
from iqrp.app.regimes.visualization.timeline import plot_timeline
from iqrp.app.regimes.visualization.transitions import plot_transitions


class RegimeDetector:
    """Facade used by all downstream forecasting models."""

    def __init__(
        self,
        settings: RegimeSettings | None = None,
        *,
        store: RegimeStore | None = None,
    ) -> None:
        ensure_regime_models_loaded()
        self.settings = settings or RegimeSettings.default()
        self.trainer = RegimeTrainer(self.settings)
        self.predictor = RegimePredictor(self.settings)
        self.serializer = RegimeSerializer()
        self.store = store or RegimeStore(
            Path(self.settings.store_dir),
            duckdb_path=Path(self.settings.duckdb_path),
            compression=self.settings.storage.compression,
            register_duckdb=self.settings.storage.register_duckdb,
        )

    def available_models(self) -> list[str]:
        return get_registry().list_names()

    def describe_model(self, name: str) -> dict[str, Any]:
        return get_registry().describe(name).to_dict()

    def fit(
        self,
        frame: pl.DataFrame,
        *,
        model_name: str | None = None,
        feature_columns: list[str] | None = None,
        artifact_path: Path | None = None,
        **model_kwargs: object,
    ) -> RegimeModel:
        return self.trainer.train(
            frame,
            model_name=model_name,
            feature_columns=feature_columns,
            artifact_path=artifact_path,
            **model_kwargs,
        )

    def detect(
        self,
        frame: pl.DataFrame,
        *,
        model: RegimeModel | None = None,
        model_name: str | None = None,
        feature_columns: list[str] | None = None,
        forecast_steps: int = 5,
        persist: bool = False,
        exchange: str = "unknown",
        symbol: str = "unknown",
        timeframe: str = "unknown",
        write_charts: bool = False,
        fit: bool = True,
    ) -> RegimeResult:
        """Fit (optional) and run full regime detection."""
        if model is None:
            if fit:
                model = self.fit(frame, model_name=model_name, feature_columns=feature_columns)
            else:
                model = get_registry().create(model_name or self.settings.default_model)
                if not model.is_fitted:
                    model = self.fit(frame, model_name=model_name, feature_columns=feature_columns)
        result = model.detect(frame, feature_columns, forecast_steps=forecast_steps)
        if persist:
            self.store.write_result(
                result,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
            )
        if write_charts and self.settings.visualization.enabled:
            out = Path(self.settings.output_dir) / "charts"
            out.mkdir(parents=True, exist_ok=True)
            plot_timeline(result, out / "regime_timeline.svg", self.settings)
            plot_transitions(result, out / "transition_graph.svg", self.settings)
            plot_persistence(result, out / "persistence.svg", self.settings)
            plot_probabilities(result, out / "probabilities.svg", self.settings)
            logger.info("regime_charts_written dir={}", out)
        return result

    def evaluate(
        self,
        model: RegimeModel,
        frame: pl.DataFrame,
        *,
        true_states: Any = None,
        feature_columns: list[str] | None = None,
    ) -> EvaluationReport:
        return model.evaluate(frame, true_states=true_states, feature_columns=feature_columns)

    def save(self, model: RegimeModel, path: Path) -> Path:
        return self.serializer.save(model, path)

    def load(self, path: Path, *, model_name: str | None = None) -> RegimeModel:
        name = model_name or self.settings.default_model
        cls = get_registry().get_class(name)
        return self.serializer.load(path, model_cls=cls)
