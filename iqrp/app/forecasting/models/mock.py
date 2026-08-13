"""Deterministic mock forecaster exercising the Institutional Forecasting Framework.

This is infrastructure scaffolding — not a production alpha model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.forecast_model import ForecastModel
from iqrp.app.forecasting.base.metadata import ForecastModelMeta, TrainingMetadata
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.postprocessing.intervals import residual_intervals
from iqrp.app.forecasting.preprocessing.windowing import recursive_path


@register_forecast_model
class MockForecastModel(ForecastModel):
    """Linear drift + last-value baseline used for framework validation."""

    meta = ForecastModelMeta(
        name="mock",
        version="1.0.0",
        description="Deterministic mock forecaster for framework tests",
        algorithm_family="baseline",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_proba=True,
        supports_intervals=True,
        supports_quantiles=False,
        parameters={"drift": 0.0},
    )

    def __init__(self, settings: Any | None = None, **kwargs: Any) -> None:
        super().__init__(settings=settings)
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._residual_std: float = 1.0
        self._last_target: float = 0.0
        self._feature_importances: np.ndarray | None = None
        self._class_centers: np.ndarray | None = None
        if kwargs:
            self.meta = ForecastModelMeta(
                **{**self.meta.to_dict(), "parameters": {**self.meta.parameters, **kwargs}}
            )

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> MockForecastModel:
        cols = self._resolve_feature_columns(frame, feature_columns)
        tgt = target_column
        if tgt is None and self._settings is not None:
            tgt = getattr(getattr(self._settings, "columns", None), "target", None)
        if tgt is None:
            # synthesize target from first feature
            tgt = cols[0] if cols else None
        x = self._matrix(frame, cols)
        if tgt is None or tgt not in frame.columns:
            y = x[:, 0] if x.size else np.zeros(frame.height)
        else:
            y = frame[tgt].to_numpy().astype(np.float64)
        n = min(x.shape[0], y.size)
        x, y = x[:n], y[:n]
        if n == 0:
            self._coef = np.zeros(x.shape[1] if x.ndim == 2 else 1)
            self._fitted = True
            return self
        # OLS with intercept via least squares on [1, X]
        ones = np.ones((n, 1), dtype=np.float64)
        design = np.concatenate([ones, x], axis=1)
        try:
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        except Exception:  # noqa: BLE001
            beta = np.zeros(design.shape[1])
            beta[0] = float(np.mean(y))
        self._intercept = float(beta[0])
        self._coef = np.asarray(beta[1:], dtype=np.float64)
        fitted = design @ beta
        self._residual_std = float(np.std(y - fitted)) or 1e-3
        self._last_target = float(y[-1])
        self._feature_columns = cols
        self._target_column = tgt
        self._regime_column = regime_column
        self._feature_importances = np.abs(self._coef)
        if self._feature_importances.sum() <= 0:
            self._feature_importances = np.ones(len(cols)) / max(len(cols), 1)
        # class centers for proba path (tertiles of y)
        qs = np.quantile(y, [0.33, 0.66])
        self._class_centers = np.asarray(
            [np.mean(y[y <= qs[0]]) if np.any(y <= qs[0]) else y.min(),
             np.mean(y[(y > qs[0]) & (y <= qs[1])]) if np.any((y > qs[0]) & (y <= qs[1])) else np.mean(y),
             np.mean(y[y > qs[1]]) if np.any(y > qs[1]) else y.max()],
            dtype=np.float64,
        )
        self._training_meta = TrainingMetadata(
            n_samples=n,
            n_features=len(cols),
            feature_columns=tuple(cols),
            target_column=tgt,
            regime_column=regime_column,
            horizon=self.meta.default_horizon,
        )
        self._fitted = True
        return self

    def partial_fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> MockForecastModel:
        if not self._fitted:
            return self.fit(
                frame, feature_columns, target_column=target_column, regime_column=regime_column
            )
        # blend with a refit on the new batch
        prev_coef = None if self._coef is None else self._coef.copy()
        prev_intercept = self._intercept
        self.fit(frame, feature_columns or self._feature_columns,
                 target_column=target_column or self._target_column,
                 regime_column=regime_column or self._regime_column)
        if prev_coef is not None and self._coef is not None and prev_coef.shape == self._coef.shape:
            self._coef = 0.7 * prev_coef + 0.3 * self._coef
            self._intercept = 0.7 * prev_intercept + 0.3 * self._intercept
        return self

    def predict(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        x = self._matrix(frame, feature_columns or self._feature_columns)
        coef = self._coef if self._coef is not None else np.zeros(x.shape[1])
        if coef.size != x.shape[1]:
            coef = np.resize(coef, x.shape[1])
        return self._intercept + x @ coef

    def predict_proba(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        preds = self.predict(frame, feature_columns)
        centers = self._class_centers
        if centers is None:
            centers = np.array([-1.0, 0.0, 1.0])
        # softmax over negative distance to centers
        logits = -np.abs(preds.reshape(-1, 1) - centers.reshape(1, -1))
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-300, None)

    def forecast(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        feature_columns: list[str] | None = None,
    ) -> Forecast:
        self._require_fitted()
        h = int(horizon if horizon is not None else self.meta.default_horizon)
        cols = feature_columns or self._feature_columns
        last_pred = float(self.predict(frame, cols)[-1]) if frame.height else self._last_target
        drift = float(self.meta.parameters.get("drift", 0.0))
        # recursive: walk last feature window forward
        x = self._matrix(frame, cols)
        if x.shape[0] >= 1:

            def step_fn(window: np.ndarray) -> float:
                row = window[-1]
                coef = self._coef if self._coef is not None else np.zeros(row.size)
                if coef.size != row.size:
                    coef = np.resize(coef, row.size)
                return float(self._intercept + row @ coef)

            if x.shape[0] >= 2:
                values = recursive_path(x[-min(8, x.shape[0]) :], step_fn, horizon=h)
            else:
                values = last_pred + drift * np.arange(1, h + 1, dtype=np.float64)
        else:
            values = last_pred + drift * np.arange(1, h + 1, dtype=np.float64)
        intervals = residual_intervals(values, residual_std=self._residual_std, level=0.95)
        regime = None
        if self._regime_column and self._regime_column in frame.columns:
            regime = frame[self._regime_column][-1]
        return Forecast.from_values(
            values,
            horizon=h,
            model_name=self.meta.name,
            model_version=self.meta.version,
            features_used=tuple(cols),
            regime_used=regime,
            strategy="recursive",
            intervals=intervals,
            metadata={"residual_std": self._residual_std},
        )

    def shap_values(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
        self._require_fitted()
        x = self._matrix(frame, feature_columns or self._feature_columns)
        coef = self._coef if self._coef is not None else np.ones(x.shape[1])
        if coef.size != x.shape[1]:
            coef = np.resize(coef, x.shape[1])
        return x * coef.reshape(1, -1)

    def integrated_gradients(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        steps: int = 16,
    ) -> np.ndarray:
        self._require_fitted()
        x = self._matrix(frame, feature_columns or self._feature_columns)
        baseline = np.zeros_like(x)
        total = np.zeros_like(x)
        for s in range(1, max(steps, 1) + 1):
            point = baseline + (s / steps) * (x - baseline)
            coef = self._coef if self._coef is not None else np.ones(point.shape[1])
            if coef.size != point.shape[1]:
                coef = np.resize(coef, point.shape[1])
            total += point * coef.reshape(1, -1)
        return (x - baseline) * total / max(steps, 1)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return self._feature_importances

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "coef": None if self._coef is None else self._coef.tolist(),
            "intercept": self._intercept,
            "residual_std": self._residual_std,
            "last_target": self._last_target,
            "feature_importances": (
                None if self._feature_importances is None else self._feature_importances.tolist()
            ),
            "class_centers": None if self._class_centers is None else self._class_centers.tolist(),
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        coef = state.get("coef")
        self._coef = None if coef is None else np.asarray(coef, dtype=np.float64)
        self._intercept = float(state.get("intercept", 0.0))
        self._residual_std = float(state.get("residual_std", 1.0))
        self._last_target = float(state.get("last_target", 0.0))
        fi = state.get("feature_importances")
        self._feature_importances = None if fi is None else np.asarray(fi, dtype=np.float64)
        cc = state.get("class_centers")
        self._class_centers = None if cc is None else np.asarray(cc, dtype=np.float64)
