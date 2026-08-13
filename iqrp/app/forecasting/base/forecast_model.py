"""Abstract forecasting model contract for all future algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.evaluator import EvaluationReport, ForecastEvaluator
from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.metadata import ForecastModelMeta, TrainingMetadata
from iqrp.app.forecasting.base.prediction import PredictionInterval
from iqrp.app.forecasting.explainability.importance import ExplanationResult, explain_model


class ForecastModel(ABC):
    """Interchangeable forecasting algorithm interface.

    Downstream code must depend only on this contract — never on concrete
    ARIMA / LSTM / gradient-boosting implementations.
    """

    meta: ForecastModelMeta

    def __init__(self, settings: Any | None = None) -> None:
        self._fitted: bool = False
        self._settings = settings
        self._feature_columns: list[str] = []
        self._target_column: str | None = None
        self._regime_column: str | None = None
        self._training_meta: TrainingMetadata | None = None
        self._checkpoint: dict[str, Any] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def settings(self) -> Any | None:
        return self._settings

    def _require_fitted(self) -> None:
        if not self._fitted:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                f"Forecast model '{self.meta.name}' is not fitted",
                code="FC_NOT_FITTED",
            )

    @abstractmethod
    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> ForecastModel:
        """Fit the forecasting model on ``frame``."""

    def partial_fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> ForecastModel:
        """Incremental update; default warm-starts by calling :meth:`fit`."""
        return self.fit(
            frame,
            feature_columns,
            target_column=target_column,
            regime_column=regime_column,
        )

    @abstractmethod
    def predict(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        """In-sample / one-step point predictions, shape ``(T,)`` or ``(T, H)``."""

    def predict_proba(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        """Class / distribution probabilities. Default: not supported."""
        self._require_fitted()
        if not self.meta.supports_proba:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                f"Model '{self.meta.name}' does not support predict_proba",
                code="FC_NO_PROBA",
            )
        raise NotImplementedError(f"{self.meta.name}.predict_proba")

    @abstractmethod
    def forecast(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        feature_columns: list[str] | None = None,
    ) -> Forecast:
        """Multi-step ahead forecast object."""

    def forecast_interval(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        level: float = 0.95,
        feature_columns: list[str] | None = None,
    ) -> list[PredictionInterval]:
        """Prediction intervals for each horizon step."""
        self._require_fitted()
        fc = self.forecast(frame, horizon=horizon, feature_columns=feature_columns)
        if fc.intervals is not None:
            return fc.intervals
        from iqrp.app.forecasting.postprocessing.intervals import residual_intervals

        preds = fc.path()
        # residual-scale fallback using recent prediction errors if available
        return residual_intervals(preds, level=level)

    def evaluate(
        self,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        probabilities: np.ndarray | None = None,
    ) -> EvaluationReport:
        self._require_fitted()
        cols = feature_columns or self._feature_columns
        tgt = target_column or self._target_column
        if tgt is None or tgt not in frame.columns:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("Target column required for evaluate", code="FC_NO_TARGET")
        y_true = frame[tgt].to_numpy()
        y_pred = self.predict(frame, cols)
        scores = None
        proba = probabilities
        if proba is None and self.meta.supports_proba:
            try:
                proba = self.predict_proba(frame, cols)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    scores = proba[:, -1]
            except Exception:  # noqa: BLE001
                proba = None
        return ForecastEvaluator().evaluate(
            y_true,
            y_pred,
            task=self.meta.task,
            probabilities=proba,
            scores=scores,
        )

    def explain(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        method: str = "permutation",
    ) -> ExplanationResult:
        self._require_fitted()
        cols = feature_columns or self._feature_columns
        return explain_model(self, frame, cols, method=method)

    def save(self, path: str | Path) -> Path:
        from iqrp.app.forecasting.serialization.serializer import ForecastSerializer

        return ForecastSerializer().save(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> ForecastModel:
        from iqrp.app.forecasting.serialization.serializer import ForecastSerializer

        return ForecastSerializer().load(Path(path), model_cls=cls)

    def get_params(self) -> dict[str, Any]:
        return dict(self.meta.parameters)

    def checkpoint(self) -> dict[str, Any]:
        """Serializable warm-start snapshot for online recovery."""
        self._require_fitted()
        payload = self.export_state()
        self._checkpoint = payload
        return payload

    def restore_checkpoint(self, payload: dict[str, Any] | None = None) -> ForecastModel:
        data = payload if payload is not None else self._checkpoint
        if data is None:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("No checkpoint available", code="FC_NO_CHECKPOINT")
        self.import_state(data)
        return self

    def export_state(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "meta": self.meta.to_dict(),
            "fitted": self._fitted,
            "feature_columns": list(self._feature_columns),
            "target_column": self._target_column,
            "regime_column": self._regime_column,
            "training_meta": None if self._training_meta is None else self._training_meta.to_dict(),
            "algorithm_state": self._algorithm_state(),
        }

    def import_state(self, payload: dict[str, Any]) -> None:
        self._fitted = bool(payload.get("fitted", False))
        self._feature_columns = list(payload.get("feature_columns") or [])
        self._target_column = payload.get("target_column")
        self._regime_column = payload.get("regime_column")
        tm = payload.get("training_meta")
        self._training_meta = None if tm is None else TrainingMetadata.from_dict(tm)
        self._load_algorithm_state(payload.get("algorithm_state") or {})

    @abstractmethod
    def _algorithm_state(self) -> dict[str, Any]:
        """Algorithm-specific fitted parameters."""

    @abstractmethod
    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        """Restore algorithm-specific fitted parameters."""

    def _resolve_feature_columns(
        self, frame: pl.DataFrame, feature_columns: list[str] | None
    ) -> list[str]:
        if feature_columns:
            return list(feature_columns)
        if self._feature_columns:
            return list(self._feature_columns)
        if self._settings is not None:
            fc = getattr(getattr(self._settings, "columns", None), "feature_columns", None)
            if fc:
                return list(fc)
        # numeric columns excluding known targets / timestamps
        exclude = {self._target_column, "open_time", "timestamp"}
        return [
            c
            for c in frame.columns
            if c not in exclude and getattr(frame[c].dtype, "is_numeric", lambda: False)()
        ]

    def _matrix(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        cols = self._resolve_feature_columns(frame, feature_columns)
        if not cols:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("No feature columns available", code="FC_NO_FEATURES")
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                f"Missing feature columns: {missing}",
                code="FC_MISSING_COLUMNS",
                details={"missing": missing},
            )
        return frame.select(cols).to_numpy().astype(np.float64)
