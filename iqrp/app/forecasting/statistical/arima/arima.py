"""ARIMA(p,d,q) forecasting model."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.statistical.base.fitting import (
    arma_innovations,
    fit_arma_css,
    forecast_arma,
    information_criteria,
)
from iqrp.app.forecasting.statistical.base.selection import select_arima_order
from iqrp.app.forecasting.statistical.base.stationarity import (
    difference,
    integrate,
    suggest_differencing,
)
from iqrp.app.forecasting.statistical.base.statistical_model import StatisticalForecastModel


@register_forecast_model
class ARIMAModel(StatisticalForecastModel):
    meta = ForecastModelMeta(
        name="arima",
        version="1.0.0",
        description="ARIMA(p,d,q) model",
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
        p: int | None = None,
        d: int | None = None,
        q: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(settings=settings, **kwargs)
        self._p, self._d, self._q = p, d, q
        self._phi = np.array([], dtype=np.float64)
        self._theta = np.array([], dtype=np.float64)
        self._intercept = 0.0

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> ARIMAModel:
        tgt = self._resolve_target_name(frame, target_column)
        y = frame[tgt].to_numpy().astype(np.float64)
        regimes = self._maybe_regime_series(frame, regime_column)
        y_fit = self._regime_conditioned_y(y, regimes)
        p, d, q = self._p, self._d, self._q
        if self._stat_settings.identification.auto and (p is None or d is None or q is None):
            sel = select_arima_order(
                y_fit,
                max_p=self._stat_settings.order.max_p,
                max_d=self._stat_settings.order.max_d,
                max_q=self._stat_settings.order.max_q,
                d=d,
                criterion=self._stat_settings.identification.criterion,  # type: ignore[arg-type]
                parallel=self._stat_settings.forecast.parallel_selection,
            )
            p = int(p if p is not None else sel.best_order.get("p", 1))
            d = int(d if d is not None else sel.best_order.get("d", 0))
            q = int(q if q is not None else sel.best_order.get("q", 0))
        else:
            d = int(
                d
                if d is not None
                else suggest_differencing(y_fit, max_d=self._stat_settings.order.max_d)
            )
            p = int(p if p is not None else (self._stat_settings.order.p or 1))
            q = int(q if q is not None else (self._stat_settings.order.q or 0))
        z = difference(y_fit, order=d) if d else y_fit
        fit = fit_arma_css(z, int(p), int(q), intercept=True)
        self._p, self._d, self._q = int(p), int(d), int(q)
        self._phi = np.asarray(fit.metadata.get("ar") or [], dtype=np.float64)
        self._theta = np.asarray(fit.metadata.get("ma") or [], dtype=np.float64)
        self._intercept = fit.intercept
        # reconstruct fitted levels
        if d:
            fitted_diff = fit.fitted
            fitted = integrate(fitted_diff, y_fit, order=d)
            # pad to length of y
            if fitted.size < y.size:
                fitted = np.concatenate([y[: y.size - fitted.size], fitted])
            resid = y - fitted[-y.size :]
        else:
            fitted = fit.fitted
            resid = fit.residuals
            if fitted.size < y.size:
                fitted = np.concatenate([y[: y.size - fitted.size], fitted])
                resid = y - fitted
        ic = information_criteria(fit.loglik, fit.k_params, fit.nobs)
        self._finalize_fit(
            y,
            target_column=tgt,
            feature_columns=feature_columns or [tgt],
            residuals=resid[-y.size :],
            fitted=fitted[-y.size :],
            sigma2=fit.sigma2,
            order={"p": self._p, "d": self._d, "q": self._q},
            ic=ic,
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
        if self._fitted_values is not None and self._fitted_values.size == y.size:
            return self._fitted_values.copy()
        z = difference(y, order=self._d or 0) if self._d else y
        _, fitted_z = arma_innovations(z, self._phi, self._theta, intercept=self._intercept)
        if self._d:
            fitted = integrate(fitted_z, y, order=int(self._d))
            if fitted.size < y.size:  # pragma: no cover
                fitted = np.concatenate([y[: y.size - fitted.size], fitted])
            return fitted[-y.size :]
        if fitted_z.size < y.size:  # pragma: no cover
            fitted_z = np.concatenate([y[: y.size - fitted_z.size], fitted_z])
        return fitted_z[-y.size :]

    def forecast(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        feature_columns: list[str] | None = None,
    ) -> Forecast:
        self._require_fitted()
        h = self._default_horizon(horizon)
        assert self._y is not None
        d = int(self._d or 0)
        z = difference(self._y, order=d) if d else self._y
        # residuals on differenced scale
        e, _ = arma_innovations(z, self._phi, self._theta, intercept=self._intercept)
        z_path = forecast_arma(z, e, self._phi, self._theta, intercept=self._intercept, horizon=h)
        path = integrate(z_path, self._y, order=d) if d else z_path
        regime = (
            frame[self._regime_column][-1]
            if self._regime_column and self._regime_column in frame.columns
            else None
        )
        return self._build_forecast(path, horizon=h, regime_used=regime)

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "p": self._p,
            "d": self._d,
            "q": self._q,
            "phi": self._phi.tolist(),
            "theta": self._theta.tolist(),
            "intercept": self._intercept,
            "y": None if self._y is None else self._y.tolist(),
            "residuals": None if self._residuals is None else self._residuals.tolist(),
            "fitted": None if self._fitted_values is None else self._fitted_values.tolist(),
            "sigma2": self._sigma2,
            "order": self._order,
            "ic": self._ic,
            "target_column": self._target_column,
            "feature_columns": self._feature_columns,
            "regime_column": self._regime_column,
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        self._p = state.get("p")
        self._d = state.get("d")
        self._q = state.get("q")
        self._phi = np.asarray(state.get("phi") or [], dtype=np.float64)
        self._theta = np.asarray(state.get("theta") or [], dtype=np.float64)
        self._intercept = float(state.get("intercept", 0.0))
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
        self._ic = dict(state.get("ic") or {})
        self._target_column = state.get("target_column")
        self._feature_columns = list(state.get("feature_columns") or [])
        self._regime_column = state.get("regime_column")
