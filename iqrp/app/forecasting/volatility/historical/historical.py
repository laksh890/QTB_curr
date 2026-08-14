"""Historical and rolling volatility estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.volatility.base.univariate import UnivariateVolatilityModel
from iqrp.app.forecasting.volatility.evaluation.metrics import realized_volatility


@register_forecast_model
class HistoricalVolatilityModel(UnivariateVolatilityModel):
    meta = ForecastModelMeta(
        name="historical_volatility",
        version="1.0.0",
        description="Full-sample historical volatility",
        algorithm_family="volatility",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_intervals=True,
    )

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> HistoricalVolatilityModel:
        tgt = self._resolve_target_name(frame, target_column)
        r = self._demean(frame[tgt].to_numpy().astype(np.float64))
        self._maybe_regime(frame, regime_column)
        ann = float(self._vol_settings.order.annualization)
        sigma = float(np.std(r, ddof=1)) if r.size > 1 else float(np.abs(r[0]) if r.size else 1e-4)
        var = np.full(r.size, max(sigma**2, 1e-12))
        params = {"sigma": sigma, "annualized": sigma * np.sqrt(ann)}
        ll = float(-0.5 * np.sum(np.log(2 * np.pi * var) + r**2 / var))
        self._finalize(
            r,
            var,
            target_column=tgt,
            params=params,
            loglik=ll,
            aic=-2 * ll + 2,
            bic=-2 * ll + np.log(r.size),
        )
        return self

    def _forecast_path(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._variance is not None
        v = float(self._variance[-1])
        var = np.full(horizon, v)
        return np.sqrt(var), var

    def _variance_from_returns(self, returns: np.ndarray) -> np.ndarray:
        s2 = float(np.var(returns, ddof=1)) if returns.size > 1 else float(returns[0] ** 2)
        return np.full(returns.size, max(s2, 1e-12))


@register_forecast_model
class RollingVolatilityModel(UnivariateVolatilityModel):
    meta = ForecastModelMeta(
        name="rolling_volatility",
        version="1.0.0",
        description="Rolling-window realized volatility",
        algorithm_family="volatility",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_intervals=True,
    )

    def __init__(
        self, settings: Any | None = None, *, window: int | None = None, **params: Any
    ) -> None:
        super().__init__(settings=settings, **params)
        self._window = window

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> RollingVolatilityModel:
        tgt = self._resolve_target_name(frame, target_column)
        r = self._demean(frame[tgt].to_numpy().astype(np.float64))
        self._maybe_regime(frame, regime_column)
        w = int(self._window or self._vol_settings.order.rolling_window)
        ann = float(self._vol_settings.order.annualization)
        # variance (not annualized) for conditional variance API
        var = np.empty(r.size)
        for t in range(r.size):
            lo = max(0, t - w + 1)
            var[t] = max(float(np.mean(r[lo : t + 1] ** 2)), 1e-12)
        rv = realized_volatility(r, window=w, annualization=ann)
        params = {"window": float(w), "last_ann_vol": float(rv[-1])}
        ll = float(-0.5 * np.sum(np.log(2 * np.pi * var) + r**2 / var))
        self._finalize(
            r,
            var,
            target_column=tgt,
            params=params,
            loglik=ll,
            aic=-2 * ll + 2,
            bic=-2 * ll + np.log(max(r.size, 1)),
            extras={"window": w},
        )
        return self

    def _forecast_path(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._variance is not None
        v = float(self._variance[-1])
        var = np.full(horizon, v)
        return np.sqrt(var), var

    def _variance_from_returns(self, returns: np.ndarray) -> np.ndarray:
        w = int(self._params.get("window", self._vol_settings.order.rolling_window))
        var = np.empty(returns.size)
        for t in range(returns.size):
            lo = max(0, t - w + 1)
            var[t] = max(float(np.mean(returns[lo : t + 1] ** 2)), 1e-12)
        return var

    def _algorithm_state(self) -> dict[str, Any]:
        state = super()._algorithm_state()
        state["window"] = self._window
        return state

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        super()._load_algorithm_state(state)
        self._window = state.get("window")
