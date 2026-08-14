"""Training orchestration for statistical forecasting models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from iqrp.app.forecasting.base.evaluator import EvaluationReport
from iqrp.app.forecasting.statistical.base.selection import SelectionResult, select_arima_order
from iqrp.app.forecasting.statistical.base.statistical_model import StatisticalForecastModel
from iqrp.app.forecasting.statistical.config import StatisticalSettings
from iqrp.app.forecasting.statistical.evaluation.metrics import evaluate_forecast
from iqrp.app.forecasting.statistical.registry import (
    create_statistical_model,
    ensure_statistical_models_loaded,
)


@dataclass(slots=True)
class StatisticalTrainResult:
    model_name: str
    order: dict[str, int]
    metrics: dict[str, float]
    selection: SelectionResult | None = None
    evaluation: EvaluationReport | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "order": dict(self.order),
            "metrics": dict(self.metrics),
            "selection": None if self.selection is None else self.selection.to_dict(),
            "evaluation": None if self.evaluation is None else self.evaluation.to_dict(),
            "diagnostics": dict(self.diagnostics),
            "metadata": dict(self.metadata),
        }


class StatisticalTrainer:
    def __init__(self, settings: StatisticalSettings | None = None) -> None:
        self.settings = settings or StatisticalSettings.default()
        ensure_statistical_models_loaded(self.settings.discovery_modules)

    def fit(
        self,
        model_name: str,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        regime_column: str | None = None,
        model: StatisticalForecastModel | None = None,
    ) -> tuple[StatisticalForecastModel, StatisticalTrainResult]:
        m = model or create_statistical_model(model_name, settings=self.settings)
        m.fit(
            frame,
            feature_columns,
            target_column=target_column or self.settings.columns.target,
            regime_column=regime_column or self.settings.regime.column,
        )
        tgt = target_column or self.settings.columns.target
        metrics: dict[str, float] = {}
        if tgt in frame.columns:
            metrics = evaluate_forecast(frame[tgt].to_numpy(), m.predict(frame))
        diag = m.diagnostics().to_dict()
        result = StatisticalTrainResult(
            model_name=m.meta.name,
            order=m.order,
            metrics=metrics,
            diagnostics=diag,
        )
        return m, result

    def auto_arima(
        self,
        frame: pl.DataFrame,
        *,
        target_column: str | None = None,
    ) -> tuple[StatisticalForecastModel, StatisticalTrainResult]:
        tgt = target_column or self.settings.columns.target
        y = frame[tgt].to_numpy()
        sel = select_arima_order(
            y,
            max_p=self.settings.order.max_p,
            max_d=self.settings.order.max_d,
            max_q=self.settings.order.max_q,
            criterion=self.settings.identification.criterion,  # type: ignore[arg-type]
            parallel=self.settings.forecast.parallel_selection,
        )
        from iqrp.app.forecasting.statistical.arima.arima import ARIMAModel

        model = ARIMAModel(
            settings=self.settings,
            p=sel.best_order.get("p"),
            d=sel.best_order.get("d"),
            q=sel.best_order.get("q"),
        )
        model.fit(frame, target_column=tgt)
        metrics = evaluate_forecast(y, model.predict(frame))
        return model, StatisticalTrainResult(
            model_name="arima",
            order=model.order,
            metrics=metrics,
            selection=sel,
            diagnostics=model.diagnostics().to_dict(),
        )

    def compare(
        self,
        model_names: list[str],
        frame: pl.DataFrame,
        *,
        target_column: str | None = None,
    ) -> list[StatisticalTrainResult]:
        rows: list[StatisticalTrainResult] = []
        for name in model_names:
            _, res = self.fit(name, frame, target_column=target_column)
            rows.append(res)
        rows.sort(key=lambda r: r.metrics.get("rmse", float("inf")))
        return rows
