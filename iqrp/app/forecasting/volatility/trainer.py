"""Training orchestration for volatility forecasting models."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from iqrp.app.forecasting.base.evaluator import EvaluationReport
from iqrp.app.forecasting.volatility.base.selection import (
    VolSelectionResult,
    select_volatility_models,
)
from iqrp.app.forecasting.volatility.base.volatility_model import VolatilityModel
from iqrp.app.forecasting.volatility.config import VolatilitySettings
from iqrp.app.forecasting.volatility.evaluation.metrics import evaluate_volatility
from iqrp.app.forecasting.volatility.registry import (
    create_volatility_model,
    ensure_volatility_models_loaded,
)
from iqrp.app.forecasting.volatility.visualization.plots import (
    plot_conditional_variance,
    plot_persistence,
    plot_residuals,
    plot_volatility_forecast,
)


@dataclass(slots=True)
class VolatilityTrainResult:
    model_name: str
    params: dict[str, float]
    metrics: dict[str, float]
    selection: VolSelectionResult | None = None
    evaluation: EvaluationReport | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    plots: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "params": dict(self.params),
            "metrics": dict(self.metrics),
            "selection": None if self.selection is None else self.selection.to_dict(),
            "evaluation": None if self.evaluation is None else self.evaluation.to_dict(),
            "diagnostics": dict(self.diagnostics),
            "metadata": dict(self.metadata),
        }


class VolatilityTrainer:
    def __init__(self, settings: VolatilitySettings | None = None) -> None:
        self.settings = settings or VolatilitySettings.default()
        ensure_volatility_models_loaded(self.settings.discovery_modules)

    def fit(
        self,
        model_name: str,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        regime_column: str | None = None,
        model: VolatilityModel | None = None,
    ) -> tuple[VolatilityModel, VolatilityTrainResult]:
        m = model or create_volatility_model(model_name, settings=self.settings)
        m.fit(
            frame,
            feature_columns,
            target_column=target_column or self.settings.columns.target,
            regime_column=regime_column
            or (self.settings.regime.column if self.settings.regime.enabled else None),
        )
        tgt = target_column or self.settings.columns.target
        metrics: dict[str, float] = {}
        if tgt in frame.columns:
            metrics = evaluate_volatility(frame[tgt].to_numpy(), m.conditional_variance())
        diag = m.diagnostics().to_dict()
        plots: dict[str, Any] = {}
        if self.settings.visualization.enabled:
            plots["forecast"] = plot_volatility_forecast(
                m.conditional_volatility(),
                forecast=m.forecast(frame).path(),
                max_points=self.settings.visualization.max_points,
            )
            plots["variance"] = plot_conditional_variance(
                m.conditional_variance(), max_points=self.settings.visualization.max_points
            )
            plots["residuals"] = plot_residuals(diag["standardized_residuals"])
            plots["persistence"] = plot_persistence(diag["persistence"], diag["half_life"])
        result = VolatilityTrainResult(
            model_name=m.meta.name,
            params=m.params,
            metrics=metrics,
            diagnostics=diag,
            plots=plots,
            metadata={"ic": m.information_criteria},
        )
        return m, result

    def auto_select(
        self,
        frame: pl.DataFrame,
        *,
        target_column: str | None = None,
        candidates: list[str] | None = None,
    ) -> tuple[VolatilityModel, VolatilityTrainResult]:
        tgt = target_column or self.settings.columns.target
        sel = select_volatility_models(
            frame,
            candidates=candidates,
            target_column=tgt,
            criterion=self.settings.selection_criterion,  # type: ignore[arg-type]
            settings=self.settings,
            parallel=True,
        )
        model, result = self.fit(sel.best, frame, target_column=tgt)
        result.selection = sel
        return model, result

    def compare(
        self,
        model_names: list[str],
        frame: pl.DataFrame,
        *,
        target_column: str | None = None,
        parallel: bool = True,
    ) -> list[VolatilityTrainResult]:
        rows: list[VolatilityTrainResult] = []

        def _one(name: str) -> VolatilityTrainResult:
            _, res = self.fit(name, frame, target_column=target_column)
            return res

        if parallel and len(model_names) > 1:
            with ThreadPoolExecutor(max_workers=min(8, len(model_names))) as pool:
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
        rows.sort(key=lambda r: r.metrics.get("qlike", float("inf")))
        return rows
