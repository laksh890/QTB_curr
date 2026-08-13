"""Holt-Winters additive seasonal exponential smoothing."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.statistical.base.statistical_model import StatisticalForecastModel


def _hw(
    y: np.ndarray, alpha: float, beta: float, gamma: float, period: int
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    s = max(int(period), 2)
    n = x.size
    level = np.empty(n)
    trend = np.empty(n)
    season = np.zeros(n + s)
    fitted = np.empty(n)
    # init
    level[0] = float(np.mean(x[:s])) if n >= s else x[0]
    trend[0] = (float(np.mean(x[s : 2 * s]) - np.mean(x[:s])) / s) if n >= 2 * s else 0.0
    for i in range(s):
        season[i] = x[i] - level[0] if i < n else 0.0
    fitted[0] = level[0] + season[0]
    for t in range(1, n):
        fitted[t] = level[t - 1] + trend[t - 1] + season[t]
        level[t] = alpha * (x[t] - season[t]) + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        season[t + s] = gamma * (x[t] - level[t]) + (1 - gamma) * season[t]
    seas_last = season[n : n + s]
    return fitted, x - fitted, float(level[-1]), float(trend[-1]), seas_last


@register_forecast_model
class HoltWintersModel(StatisticalForecastModel):
    meta = ForecastModelMeta(
        name="holt_winters",
        version="1.0.0",
        description="Holt-Winters additive seasonal smoothing",
        algorithm_family="statistical",
        task="regression",
        default_horizon=12,
        supports_online=True,
        supports_intervals=True,
    )

    def __init__(
        self,
        settings: Any | None = None,
        *,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        seasonal_period: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(settings=settings, **kwargs)
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._s = seasonal_period
        self._level = 0.0
        self._trend = 0.0
        self._season = np.zeros(1)

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> HoltWintersModel:
        tgt = self._resolve_target_name(frame, target_column)
        y = frame[tgt].to_numpy().astype(np.float64)
        self._maybe_regime_series(frame, regime_column)
        s = int(self._s or self._stat_settings.order.seasonal_period)
        self._s = s
        if self._alpha is None or self._beta is None or self._gamma is None:
            def obj(theta: np.ndarray) -> float:
                _, r, _, _, _ = _hw(y, float(theta[0]), float(theta[1]), float(theta[2]), s)
                return float(np.dot(r, r))

            res = minimize(
                obj,
                x0=np.array([0.3, 0.1, 0.1]),
                bounds=[(0.01, 0.99)] * 3,
                method="L-BFGS-B",
            )
            a, b, g = res.x if res.success else np.array([0.3, 0.1, 0.1])
            self._alpha, self._beta, self._gamma = float(a), float(b), float(g)
        fitted, resid, lvl, tr, seas = _hw(
            y, float(self._alpha), float(self._beta), float(self._gamma), s
        )
        self._level, self._trend, self._season = lvl, tr, seas
        self._finalize_fit(
            y,
            target_column=tgt,
            feature_columns=feature_columns or [tgt],
            residuals=resid,
            fitted=fitted,
            sigma2=float(np.var(resid)) or 1e-12,
            order={"s": s},
            algorithm_extras={
                "alpha": self._alpha,
                "beta": self._beta,
                "gamma": self._gamma,
            },
        )
        return self

    def predict(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        tgt = self._target_column or self._stat_settings.columns.target
        y = frame[tgt].to_numpy().astype(np.float64) if tgt in frame.columns else self._extract_target(frame, None)
        fitted, _, _, _, _ = _hw(
            y, float(self._alpha or 0.3), float(self._beta or 0.1), float(self._gamma or 0.1), int(self._s or 12)
        )
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
        s = int(self._s or 12)
        path = np.empty(h, dtype=np.float64)
        for i in range(h):
            path[i] = self._level + (i + 1) * self._trend + float(self._season[i % s])
        regime = frame[self._regime_column][-1] if self._regime_column and self._regime_column in frame.columns else None
        return self._build_forecast(path, horizon=h, strategy="direct", regime_used=regime)

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "alpha": self._alpha,
            "beta": self._beta,
            "gamma": self._gamma,
            "s": self._s,
            "level": self._level,
            "trend": self._trend,
            "season": self._season.tolist(),
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
        self._gamma = state.get("gamma")
        self._s = state.get("s")
        self._level = float(state.get("level", 0.0))
        self._trend = float(state.get("trend", 0.0))
        self._season = np.asarray(state.get("season") or [0.0], dtype=np.float64)
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
