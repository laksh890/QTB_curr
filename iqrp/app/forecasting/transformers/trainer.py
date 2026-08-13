"""High-level orchestration for transformer forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from iqrp.app.forecasting.base.evaluator import EvaluationReport
from iqrp.app.forecasting.transformers.base.transformer_model import TransformerForecastModel
from iqrp.app.forecasting.transformers.config import TransformerSettings
from iqrp.app.forecasting.transformers.diagnostics.report import run_transformer_diagnostics
from iqrp.app.forecasting.transformers.registry import create_transformer_model, ensure_transformer_models_loaded
from iqrp.app.forecasting.transformers.visualization.plots import (
    plot_attention_map,
    plot_forecast,
    plot_residual_distribution,
    plot_training_curves,
)


@dataclass(slots=True)
class TransformerTrainResult:
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


class TransformerOrchestrator:
    def __init__(self, settings: TransformerSettings | None = None) -> None:
        self.settings = settings or TransformerSettings.default()
        ensure_transformer_models_loaded(self.settings.discovery_modules)

    def fit(
        self,
        model_name: str,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        regime_column: str | None = None,
        model: TransformerForecastModel | None = None,
    ) -> tuple[TransformerForecastModel, TransformerTrainResult]:
        m = model or create_transformer_model(model_name, settings=self.settings)
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
            plots["forecast"] = plot_forecast(fc.path())
            attn = m.attention()
            if attn.size > 1:
                plots["attention"] = plot_attention_map(attn)
        result = TransformerTrainResult(
            model_name=m.meta.name,
            metrics=ev.metrics,
            evaluation=ev,
            diagnostics=diag,
            plots=plots,
            metadata={"architecture": m.architecture_name, "history": m._history.to_dict()},
        )
        return m, result
