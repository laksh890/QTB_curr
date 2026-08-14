"""Base class for all institutional statistical forecasting models."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.evaluator import EvaluationReport, ForecastEvaluator
from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.forecast_model import ForecastModel
from iqrp.app.forecasting.base.metadata import TrainingMetadata
from iqrp.app.forecasting.base.prediction import PredictionInterval
from iqrp.app.forecasting.postprocessing.intervals import residual_intervals
from iqrp.app.forecasting.statistical.config import StatisticalSettings
from iqrp.app.forecasting.statistical.diagnostics.report import DiagnosticReport, run_diagnostics


class StatisticalForecastModel(ForecastModel):
    """Classical econometric forecaster with shared identification / diagnostics."""

    def __init__(
        self,
        settings: StatisticalSettings | Any | None = None,
        **params: Any,
    ) -> None:
        if settings is None:
            settings = StatisticalSettings.default()
        elif isinstance(settings, dict):
            settings = StatisticalSettings.from_mapping(settings)
        super().__init__(settings=settings)
        self._stat_settings: StatisticalSettings = settings  # type: ignore[assignment]
        self._params: dict[str, Any] = dict(params)
        self._y: np.ndarray | None = None
        self._residuals: np.ndarray | None = None
        self._fitted_values: np.ndarray | None = None
        self._sigma2: float = 1.0
        self._order: dict[str, int] = {}
        self._ic: dict[str, float] = {}
        self._regime_models: dict[Any, dict[str, Any]] = {}
        self._endog_names: list[str] = []
        self._exog: np.ndarray | None = None
        self._online_buffer: list[float] = []

    @property
    def order(self) -> dict[str, int]:
        return dict(self._order)

    @property
    def information_criteria(self) -> dict[str, float]:
        return dict(self._ic)

    def residuals(self) -> np.ndarray:
        self._require_fitted()
        if self._residuals is None:
            return np.array([], dtype=np.float64)
        return np.asarray(self._residuals, dtype=np.float64)

    def diagnostics(self) -> DiagnosticReport:
        self._require_fitted()
        return run_diagnostics(self.residuals())

    def evaluate(
        self,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        probabilities: np.ndarray | None = None,
    ) -> EvaluationReport:
        self._require_fitted()
        tgt = target_column or self._target_column or self._stat_settings.columns.target
        if tgt not in frame.columns:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("Target column required for evaluate", code="STAT_NO_TARGET")
        y_true = frame[tgt].to_numpy().astype(np.float64)
        y_pred = self.predict(frame, feature_columns)
        return ForecastEvaluator().evaluate(y_true, y_pred, task="regression")

    def forecast_interval(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        level: float | None = None,
        feature_columns: list[str] | None = None,
    ) -> list[PredictionInterval]:
        self._require_fitted()
        fc = self.forecast(frame, horizon=horizon, feature_columns=feature_columns)
        if fc.intervals is not None:
            return fc.intervals
        lvl = float(level if level is not None else self._stat_settings.forecast.interval_level)
        # grow sigma with horizon for recursive forecasts
        h = fc.path().size
        sig = self._sigma2**0.5 * np.sqrt(np.arange(1, h + 1, dtype=np.float64))
        return residual_intervals(fc.path(), residual_std=sig, level=lvl)

    def partial_fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> StatisticalForecastModel:
        mode = self._stat_settings.online.mode
        if not self._fitted or not self._stat_settings.online.warm_start:
            return self.fit(  # type: ignore[return-value]
                frame, feature_columns, target_column=target_column, regime_column=regime_column
            )
        y_new = self._extract_target(frame, target_column)
        if self._y is None:
            return self.fit(  # type: ignore[return-value]
                frame, feature_columns, target_column=target_column, regime_column=regime_column
            )
        if mode == "sliding" or mode == "rolling":
            w = int(self._stat_settings.online.window)
            y_all = np.concatenate([self._y, y_new])[-w:]
        else:  # expanding
            y_all = np.concatenate([self._y, y_new])
        # rebuild frame for refit
        rebuilt = self._frame_from_y(y_all, frame, target_column)
        return self.fit(  # type: ignore[return-value]
            rebuilt,
            feature_columns or self._feature_columns,
            target_column=target_column or self._target_column,
            regime_column=regime_column or self._regime_column,
        )

    def _extract_target(self, frame: pl.DataFrame, target_column: str | None) -> np.ndarray:
        tgt = target_column or self._target_column or self._stat_settings.columns.target
        if tgt in frame.columns:
            return frame[tgt].to_numpy().astype(np.float64)
        cols = self._resolve_feature_columns(frame, None)
        if cols:  # pragma: no cover
            return frame[cols[0]].to_numpy().astype(np.float64)
        from iqrp.app.core.exceptions import ValidationError

        raise ValidationError("Unable to extract target series", code="STAT_NO_TARGET")

    def _frame_from_y(
        self, y: np.ndarray, template: pl.DataFrame, target_column: str | None
    ) -> pl.DataFrame:
        tgt = target_column or self._target_column or self._stat_settings.columns.target
        data: dict[str, Any] = {
            self._stat_settings.columns.timestamp: list(range(len(y))),
            tgt: y,
        }
        if self._regime_column and self._regime_column in template.columns:
            # repeat last regimes / pad
            reg = template[self._regime_column].to_numpy()
            if reg.size >= y.size:
                data[self._regime_column] = reg[-y.size :]
            else:
                data[self._regime_column] = np.resize(reg, y.size)
        return pl.DataFrame(data)

    def _resolve_target_name(self, frame: pl.DataFrame, target_column: str | None) -> str:
        if target_column:
            return target_column
        if self._target_column:
            return self._target_column
        cfg = self._stat_settings.columns.target
        if cfg in frame.columns:
            return cfg
        cols = self._resolve_feature_columns(frame, None)
        if cols:
            return cols[0]
        from iqrp.app.core.exceptions import ValidationError

        raise ValidationError("No target column available", code="STAT_NO_TARGET")

    def _maybe_regime_series(
        self, frame: pl.DataFrame, regime_column: str | None
    ) -> np.ndarray | None:
        col = regime_column or (
            self._stat_settings.regime.column if self._stat_settings.regime.enabled else None
        )
        if col and col in frame.columns:
            self._regime_column = col
            return frame[col].to_numpy()
        return None

    def _regime_conditioned_y(self, y: np.ndarray, regimes: np.ndarray | None) -> np.ndarray:
        """Optionally demean by regime for conditioning."""
        if regimes is None or not self._stat_settings.regime.condition_forecasts:
            return y
        out = y.copy()
        for r in np.unique(regimes):
            mask = regimes == r
            if np.any(mask):
                out[mask] = out[mask] - np.mean(out[mask]) + np.mean(y)
        return out

    def _finalize_fit(
        self,
        y: np.ndarray,
        *,
        target_column: str,
        feature_columns: list[str] | None,
        residuals: np.ndarray,
        fitted: np.ndarray,
        sigma2: float,
        order: dict[str, int],
        ic: dict[str, float] | None = None,
        algorithm_extras: dict[str, Any] | None = None,
    ) -> None:
        self._y = np.asarray(y, dtype=np.float64).reshape(-1)
        self._residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
        self._fitted_values = np.asarray(fitted, dtype=np.float64).reshape(-1)
        self._sigma2 = max(float(sigma2), 1e-12)
        self._order = dict(order)
        self._ic = dict(ic or {})
        self._target_column = target_column
        self._feature_columns = list(feature_columns or [target_column])
        self._training_meta = TrainingMetadata(
            n_samples=int(self._y.size),
            n_features=len(self._feature_columns),
            feature_columns=tuple(self._feature_columns),
            target_column=target_column,
            regime_column=self._regime_column,
            horizon=self._stat_settings.forecast.default_horizon,
            extra={"order": self._order, "ic": self._ic, **(algorithm_extras or {})},
        )
        self._fitted = True

    def _build_forecast(
        self,
        values: np.ndarray,
        *,
        horizon: int,
        strategy: str = "recursive",
        regime_used: Any = None,
    ) -> Forecast:
        path = np.asarray(values, dtype=np.float64).reshape(-1)
        sig = self._sigma2**0.5 * np.sqrt(np.arange(1, path.size + 1, dtype=np.float64))
        intervals = residual_intervals(
            path,
            residual_std=sig,
            level=self._stat_settings.forecast.interval_level,
        )
        return Forecast.from_values(
            path,
            horizon=horizon,
            model_name=self.meta.name,
            model_version=self.meta.version,
            features_used=tuple(self._feature_columns),
            regime_used=regime_used,
            strategy=strategy,
            intervals=intervals,
            metadata={"order": self._order, "sigma2": self._sigma2, "ic": self._ic},
        )

    def _default_horizon(self, horizon: int | None) -> int:
        if horizon is not None:
            return max(int(horizon), 1)
        return max(int(self._stat_settings.forecast.default_horizon), 1)

    @abstractmethod
    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> StatisticalForecastModel: ...

    @abstractmethod
    def predict(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray: ...

    @abstractmethod
    def forecast(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        feature_columns: list[str] | None = None,
    ) -> Forecast: ...
