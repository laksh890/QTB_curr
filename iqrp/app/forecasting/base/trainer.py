"""Training orchestration helpers for forecasting models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.evaluator import EvaluationReport, ForecastEvaluator
from iqrp.app.forecasting.base.metadata import TrainingMetadata
from iqrp.app.forecasting.base.registry import get_registry


@dataclass(slots=True)
class TrainResult:
    model_name: str
    training: TrainingMetadata
    evaluation: EvaluationReport | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "training": self.training.to_dict(),
            "evaluation": None if self.evaluation is None else self.evaluation.to_dict(),
            "history": list(self.history),
            "metadata": dict(self.metadata),
        }


class ForecastTrainer:
    """Standard fit / validate / record lifecycle for any registered model."""

    def __init__(self, settings: Any | None = None) -> None:
        self.settings = settings
        self.evaluator = ForecastEvaluator()

    def resolve_columns(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None,
        target_column: str | None,
    ) -> tuple[list[str], str | None]:
        cols = list(feature_columns or [])
        if not cols and self.settings is not None:
            fc = getattr(getattr(self.settings, "columns", None), "feature_columns", None)
            if fc:
                cols = list(fc)
        if not cols:
            exclude = {target_column} if target_column else set()
            ts = getattr(getattr(self.settings, "columns", None), "timestamp", "open_time")
            exclude.add(ts)
            cols = [c for c in frame.columns if c not in exclude and frame[c].dtype.is_numeric()]
        tgt = target_column
        if tgt is None and self.settings is not None:
            tgt = getattr(getattr(self.settings, "columns", None), "target", None)
        return cols, tgt

    def build_training_metadata(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str],
        target_column: str | None,
        *,
        horizon: int = 1,
        regime_column: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> TrainingMetadata:
        ts = getattr(getattr(self.settings, "columns", None), "timestamp", "open_time")
        start = frame[ts][0] if ts in frame.columns and frame.height else None
        end = frame[ts][-1] if ts in frame.columns and frame.height else None
        return TrainingMetadata(
            n_samples=int(frame.height),
            n_features=len(feature_columns),
            feature_columns=tuple(feature_columns),
            target_column=target_column,
            regime_column=regime_column,
            horizon=horizon,
            train_start=start,
            train_end=end,
            extra=dict(extra or {}),
        )

    def fit(
        self,
        model: Any,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        regime_column: str | None = None,
        validate: bool = True,
        validation_fraction: float | None = None,
    ) -> TrainResult:
        cols, tgt = self.resolve_columns(frame, feature_columns, target_column)
        if not cols:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("No feature columns resolved for training", code="FC_NO_FEATURES")
        horizon = int(getattr(model.meta, "default_horizon", 1))
        frac = validation_fraction
        if frac is None and self.settings is not None:
            frac = float(getattr(getattr(self.settings, "training", None), "validation_fraction", 0.0))
        frac = float(frac or 0.0)
        history: list[dict[str, Any]] = []
        eval_report: EvaluationReport | None = None
        if validate and 0.0 < frac < 1.0 and frame.height >= 5:
            split = max(1, int(frame.height * (1.0 - frac)))
            train_frame = frame.slice(0, split)
            val_frame = frame.slice(split, frame.height - split)
            model.fit(train_frame, cols, target_column=tgt, regime_column=regime_column)
            if tgt and tgt in val_frame.columns:
                y_true = val_frame[tgt].to_numpy()
                y_pred = model.predict(val_frame, cols)
                eval_report = self.evaluator.evaluate(y_true, y_pred, task=model.meta.task)
                history.append({"stage": "validation", "metrics": eval_report.metrics})
        else:
            model.fit(frame, cols, target_column=tgt, regime_column=regime_column)
        meta = self.build_training_metadata(
            frame,
            cols,
            tgt,
            horizon=horizon,
            regime_column=regime_column,
        )
        get_registry().record_training(model.meta.name, meta)
        return TrainResult(
            model_name=model.meta.name,
            training=meta,
            evaluation=eval_report,
            history=history,
        )

    def partial_fit(
        self,
        model: Any,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> TrainResult:
        cols, tgt = self.resolve_columns(frame, feature_columns, target_column)
        model.partial_fit(frame, cols, target_column=tgt, regime_column=regime_column)
        meta = self.build_training_metadata(
            frame,
            cols,
            tgt,
            horizon=int(getattr(model.meta, "default_horizon", 1)),
            regime_column=regime_column,
            extra={"mode": "partial_fit"},
        )
        get_registry().record_training(model.meta.name, meta)
        return TrainResult(model_name=model.meta.name, training=meta, metadata={"mode": "partial_fit"})
