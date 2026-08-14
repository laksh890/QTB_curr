"""Training orchestration for tree-based forecasting models."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from iqrp.app.forecasting.base.evaluator import EvaluationReport
from iqrp.app.forecasting.tree_models.base.tree_model import TreeForecastModel
from iqrp.app.forecasting.tree_models.config import TreeSettings
from iqrp.app.forecasting.tree_models.diagnostics.report import run_tree_diagnostics
from iqrp.app.forecasting.tree_models.registry import create_tree_model, ensure_tree_models_loaded
from iqrp.app.forecasting.tree_models.visualization.plots import (
    plot_feature_importance,
    plot_learning_curve,
    plot_prediction_error,
    plot_residual_distribution,
)


@dataclass(slots=True)
class TreeTrainResult:
    model_name: str
    metrics: dict[str, float]
    best_params: dict[str, Any]
    evaluation: EvaluationReport | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    plots: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "metrics": dict(self.metrics),
            "best_params": dict(self.best_params),
            "evaluation": None if self.evaluation is None else self.evaluation.to_dict(),
            "diagnostics": dict(self.diagnostics),
            "metadata": dict(self.metadata),
        }


class TreeTrainer:
    def __init__(self, settings: TreeSettings | None = None) -> None:
        self.settings = settings or TreeSettings.default()
        ensure_tree_models_loaded(self.settings.discovery_modules)

    def fit(
        self,
        model_name: str,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        regime_column: str | None = None,
        model: TreeForecastModel | None = None,
    ) -> tuple[TreeForecastModel, TreeTrainResult]:
        m = model or create_tree_model(model_name, settings=self.settings)
        m.fit(
            frame,
            feature_columns,
            target_column=target_column or self.settings.columns.target,
            regime_column=regime_column
            or (self.settings.regime.column if self.settings.regime.enabled else None),
        )
        ev = m.evaluate(frame, feature_columns=m._feature_columns)

        diag = {}
        plots: dict[str, Any] = {}
        if m._X is not None and m._y is not None:
            report = run_tree_diagnostics(
                m._estimator,
                m._X,
                m._y,
                backend=m.backend,
                task=self.settings.task.type,
                params=m.best_params,
                feature_names=m._feature_columns,
            )
            diag = report.to_dict()
            if self.settings.visualization.enabled:
                plots["importance"] = plot_feature_importance(m.feature_importance())
                plots["learning_curve"] = plot_learning_curve(report.learning_curve)
                y_hat = m._train_pred if m._train_pred is not None else m._y
                resid = m._residuals if m._residuals is not None else (m._y - y_hat)
                plots["prediction_error"] = plot_prediction_error(m._y, y_hat)
                plots["residuals"] = plot_residual_distribution(resid)

        result = TreeTrainResult(
            model_name=m.meta.name,
            metrics=ev.metrics,
            best_params=m.best_params,
            evaluation=ev,
            diagnostics=diag,
            plots=plots,
            metadata={"cv_scores": m._cv_scores},
        )
        return m, result

    def compare(
        self,
        model_names: list[str],
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        parallel: bool = True,
    ) -> list[TreeTrainResult]:
        rows: list[TreeTrainResult] = []

        def _one(name: str) -> TreeTrainResult:
            _, res = self.fit(
                name, frame, feature_columns=feature_columns, target_column=target_column
            )
            return res

        if parallel and len(model_names) > 1:
            with ThreadPoolExecutor(max_workers=min(6, len(model_names))) as pool:
                futs = {pool.submit(_one, n): n for n in model_names}
                for fut in as_completed(futs):
                    try:
                        rows.append(fut.result())
                    except Exception:
                        continue
        else:
            for name in model_names:
                try:
                    rows.append(_one(name))
                except Exception:
                    continue
        rows.sort(key=lambda r: r.metrics.get("rmse", float("inf")))
        return rows
