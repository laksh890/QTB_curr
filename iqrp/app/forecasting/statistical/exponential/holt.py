"""Holt linear trend exponential smoothing."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.statistical.base.statistical_model import StatisticalForecastModel


def _holt(y: np.ndarray, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    n = x.size
    level = np.empty(n)
    trend = np.empty(n)
    fitted = np.empty(n)
    level[0] = x[0]
    trend[0] = x[1] - x[0] if n > 1 else 0.0
    fitted[0] = x[0]
    for t in range(1, n):
        fitted[t] = level[t - 1] + trend[t - 1]
        level[t] = alpha * x[t] + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
    return fitted, x - fitted, float(level[-1]), float(trend[-1])


@register_forecast_model
class HoltModel(StatisticalForecastModel):
    meta = ForecastModelMeta(
        name="holt",
        version="1.0.0",
        description="Holt linear trend exponential smoothing",
        algorithm_family="statistical",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_intervals=True,
    )

    def __init__(
        self,
        settings: Any | None = None,
        *,
        alpha: float | None = None,
        beta: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(settings=settings, **kwargs)
        self._alpha = alpha
        self._beta = beta
        self._level = 0.0
        self._trend = 0.0

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> HoltModel:
        tgt = self._resolve_target_name(frame, target_column)
        y = frame[tgt].to_numpy().astype(np.float64)
        self._maybe_regime_series(frame, regime_column)
        if self._alpha is None or self._beta is None:

            def obj(theta: np.ndarray) -> float:
                _, r, _, _ = _holt(y, float(theta[0]), float(theta[1]))
                return float(np.dot(r, r))

            res = minimize(
                obj, x0=np.array([0.3, 0.1]), bounds=[(0.01, 0.99), (0.01, 0.99)], method="L-BFGS-B"
            )
            a, b = res.x if res.success else np.array([0.3, 0.1])
            self._alpha = float(a)
            self._beta = float(b)
        fitted, resid, lvl, tr = _holt(y, float(self._alpha), float(self._beta))
        self._level, self._trend = lvl, tr
        self._finalize_fit(
            y,
            target_column=tgt,
            feature_columns=feature_columns or [tgt],
            residuals=resid,
            fitted=fitted,
            sigma2=float(np.var(resid)) or 1e-12,
            order={},
            algorithm_extras={"alpha": self._alpha, "beta": self._beta},
        )
        return self

    def predict(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
        self._require_fitted()
        tgt = self._target_column or self._stat_settings.columns.target
        y = (
            frame[tgt].to_numpy().astype(np.float64)
            if tgt in frame.columns
            else self._extract_target(frame, None)
        )
        fitted, _, _, _ = _holt(y, float(self._alpha or 0.3), float(self._beta or 0.1))
        return fitted

    def forecast(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        feature_columns: list[str] | None = None,
    ) -> Forecast:
        self._require_fitted()
        h = self._default_horizon(horizon)
        path = np.asarray([self._level + (i + 1) * self._trend for i in range(h)], dtype=np.float64)
        regime = (
            frame[self._regime_column][-1]
            if self._regime_column and self._regime_column in frame.columns
            else None
        )
        return self._build_forecast(path, horizon=h, strategy="direct", regime_used=regime)

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "alpha": self._alpha,
            "beta": self._beta,
            "level": self._level,
            "trend": self._trend,
            "y": None if self._y is None else self._y.tolist(),
            "residuals": None if self._residuals is None else self._residuals.tolist(),
            "fitted": None if self._fitted_values is None else self._fitted_values.tolist(),
            "sigma2": self._sigma2,
            "order": self._order,
            "target_column": self._target_column,
            "feature_columns": self._feature_columns,
            "regime_column": self._regime_column,
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        self._alpha = state.get("alpha")
        self._beta = state.get("beta")
        self._level = float(state.get("level", 0.0))
        self._trend = float(state.get("trend", 0.0))
        self._y = None if state.get("y") is None else np.asarray(state["y"], dtype=np.float64)
        self._residuals = (
            None
            if state.get("residuals") is None
            else np.asarray(state["residuals"], dtype=np.float64)
        )
        self._fitted_values = (
            None if state.get("fitted") is None else np.asarray(state["fitted"], dtype=np.float64)
        )
        self._sigma2 = float(state.get("sigma2", 1.0))
        self._order = dict(state.get("order") or {})
        self._target_column = state.get("target_column")
        self._feature_columns = list(state.get("feature_columns") or [])
        self._regime_column = state.get("regime_column")
