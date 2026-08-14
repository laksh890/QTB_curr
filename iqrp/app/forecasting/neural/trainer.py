"""Training orchestration for neural forecasting models."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from iqrp.app.forecasting.base.evaluator import EvaluationReport
from iqrp.app.forecasting.neural.base.neural_model import NeuralForecastModel
from iqrp.app.forecasting.neural.config import NeuralSettings
from iqrp.app.forecasting.neural.registry import create_neural_model, ensure_neural_models_loaded
from iqrp.app.forecasting.neural.visualization.plots import (
    plot_forecast,
    plot_residual_distribution,
    plot_training_curves,
)


@dataclass(slots=True)
class NeuralTrainResult:
    model_name: str
    metrics: dict[str, float]
    evaluation: EvaluationReport | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    plots: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "metrics": dict(self.metrics),
            "evaluation": None if self.evaluation is None else self.evaluation.to_dict(),
            "diagnostics": dict(self.diagnostics),
            "metadata": dict(self.metadata),
        }


class NeuralOrchestrator:
    """High-level fit / compare / evaluate for neural architectures."""

    def __init__(self, settings: NeuralSettings | None = None) -> None:
        self.settings = settings or NeuralSettings.default()
        ensure_neural_models_loaded(self.settings.discovery_modules)

    def fit(
        self,
        model_name: str,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        regime_column: str | None = None,
        model: NeuralForecastModel | None = None,
    ) -> tuple[NeuralForecastModel, NeuralTrainResult]:
        m = model or create_neural_model(model_name, settings=self.settings)
        m.fit(
            frame,
            feature_columns,
            target_column=target_column or self.settings.columns.target,
            regime_column=regime_column
            or (self.settings.regime.column if self.settings.regime.enabled else None),
        )
        ev = m.evaluate(frame, feature_columns=m._feature_columns)
        diag = m.diagnostics()
        plots: dict[str, Any] = {}
        if self.settings.visualization.enabled:
            plots["training"] = plot_training_curves(m._history.to_dict())
            if m._residuals is not None:
                plots["residuals"] = plot_residual_distribution(m._residuals)
            fc = m.forecast(frame, horizon=min(self.settings.forecast.default_horizon, m._horizon))
            plots["forecast"] = plot_forecast(None, fc.path())
        result = NeuralTrainResult(
            model_name=m.meta.name,
            metrics=ev.metrics,
            evaluation=ev,
            diagnostics=diag,
            plots=plots,
            metadata={"architecture": m.architecture_name, "history": m._history.to_dict()},
        )
        return m, result

    def compare(
        self,
        model_names: list[str],
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        parallel: bool = False,
    ) -> dict[str, NeuralTrainResult]:
        results: dict[str, NeuralTrainResult] = {}

        def _one(name: str) -> tuple[str, NeuralTrainResult]:
            _, res = self.fit(name, frame, feature_columns=feature_columns)
            return name, res

        if parallel and len(model_names) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(model_names))) as ex:
                futs = [ex.submit(_one, n) for n in model_names]
                for fut in as_completed(futs):
                    name, res = fut.result()
                    results[name] = res
        else:
            for name in model_names:
                _, res = self.fit(name, frame, feature_columns=feature_columns)
                results[name] = res
        return results


# Alias matching architecture naming
NeuralTrainerFacade = NeuralOrchestrator
