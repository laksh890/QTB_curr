"""Simple Exponential Smoothing."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.statistical.base.statistical_model import StatisticalForecastModel


def _ses(y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    level = np.empty(x.size, dtype=np.float64)
    fitted = np.empty(x.size, dtype=np.float64)
    level[0] = x[0]
    fitted[0] = x[0]
    for t in range(1, x.size):
        fitted[t] = level[t - 1]
        level[t] = alpha * x[t] + (1 - alpha) * level[t - 1]
    resid = x - fitted
    return fitted, resid


@register_forecast_model
class SimpleExpSmoothingModel(StatisticalForecastModel):
    meta = ForecastModelMeta(
        name="ses",
        version="1.0.0",
        description="Simple Exponential Smoothing",
        algorithm_family="statistical",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_intervals=True,
    )

    def __init__(self, settings: Any | None = None, *, alpha: float | None = None, **kwargs: Any) -> None:
        super().__init__(settings=settings, **kwargs)
        self._alpha = alpha
        self._level = 0.0

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> SimpleExpSmoothingModel:
        tgt = self._resolve_target_name(frame, target_column)
        y = frame[tgt].to_numpy().astype(np.float64)
        self._maybe_regime_series(frame, regime_column)
        if self._alpha is None:
            def obj(a: float) -> float:
                _, r = _ses(y, float(a))
                return float(np.dot(r, r))

            res = minimize_scalar(obj, bounds=(0.01, 0.99), method="bounded")
            self._alpha = float(res.x) if res.success else 0.2
        fitted, resid = _ses(y, float(self._alpha))
        self._level = float(fitted[-1] if fitted.size else 0.0)
        # last level after update
        a = float(self._alpha)
        lvl = y[0]
        for t in range(y.size):
            lvl = a * y[t] + (1 - a) * lvl
        self._level = float(lvl)
        self._finalize_fit(
            y,
            target_column=tgt,
            feature_columns=feature_columns or [tgt],
            residuals=resid,
            fitted=fitted,
            sigma2=float(np.var(resid)) or 1e-12,
            order={"alpha": 0},
            ic={},
            algorithm_extras={"alpha": self._alpha, "level": self._level},
        )
        self._order = {"alpha": int(round(100 * float(self._alpha)))}
        return self

    def predict(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        tgt = self._target_column or self._stat_settings.columns.target
        y = frame[tgt].to_numpy().astype(np.float64) if tgt in frame.columns else self._extract_target(frame, None)
        fitted, _ = _ses(y, float(self._alpha or 0.2))
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
        path = np.full(h, self._level, dtype=np.float64)
        regime = frame[self._regime_column][-1] if self._regime_column and self._regime_column in frame.columns else None
        return self._build_forecast(path, horizon=h, strategy="direct", regime_used=regime)

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "alpha": self._alpha,
            "level": self._level,
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
        self._level = float(state.get("level", 0.0))
        self._y = None if state.get("y") is None else np.asarray(state["y"], dtype=np.float64)
        self._residuals = (
            None if state.get("residuals") is None else np.asarray(state["residuals"], dtype=np.float64)
        )
        self._fitted_values = (
            None if state.get("fitted") is None else np.asarray(state["fitted"], dtype=np.float64)
        )
        self._sigma2 = float(state.get("sigma2", 1.0))
        self._order = dict(state.get("order") or {})
        self._target_column = state.get("target_column")
        self._feature_columns = list(state.get("feature_columns") or [])
        self._regime_column = state.get("regime_column")
