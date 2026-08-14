"""Moving Average MA(q) forecasting model."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.statistical.base.fitting import (
    fit_arma_css,
    forecast_arma,
    information_criteria,
)
from iqrp.app.forecasting.statistical.base.selection import select_arma_order
from iqrp.app.forecasting.statistical.base.statistical_model import StatisticalForecastModel


@register_forecast_model
class MAModel(StatisticalForecastModel):
    meta = ForecastModelMeta(
        name="ma",
        version="1.0.0",
        description="Moving Average MA(q) model",
        algorithm_family="statistical",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_intervals=True,
    )

    def __init__(self, settings: Any | None = None, *, q: int | None = None, **kwargs: Any) -> None:
        super().__init__(settings=settings, **kwargs)
        self._q = q
        self._theta = np.array([], dtype=np.float64)
        self._intercept = 0.0

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> MAModel:
        tgt = self._resolve_target_name(frame, target_column)
        y = frame[tgt].to_numpy().astype(np.float64)
        regimes = self._maybe_regime_series(frame, regime_column)
        y_fit = self._regime_conditioned_y(y, regimes)
        q = self._q
        if q is None:
            if self._stat_settings.order.q is not None:
                q = int(self._stat_settings.order.q)
            elif self._stat_settings.identification.auto:
                sel = select_arma_order(
                    y_fit,
                    max_p=0,
                    max_q=self._stat_settings.order.max_q,
                    criterion=self._stat_settings.identification.criterion,  # type: ignore[arg-type]
                    parallel=self._stat_settings.forecast.parallel_selection,
                )
                q = int(sel.best_order.get("q", 1))
            else:
                q = 1
        fit = fit_arma_css(y_fit, 0, int(q), intercept=True)
        self._q = int(q)
        self._theta = np.asarray(fit.metadata.get("ma") or fit.params, dtype=np.float64)
        self._intercept = fit.intercept
        ic = information_criteria(fit.loglik, fit.k_params, fit.nobs)
        self._finalize_fit(
            y,
            target_column=tgt,
            feature_columns=feature_columns or [tgt],
            residuals=fit.residuals,
            fitted=fit.fitted,
            sigma2=fit.sigma2,
            order={"q": int(q)},
            ic=ic,
            algorithm_extras={"theta": self._theta.tolist(), "intercept": self._intercept},
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
        from iqrp.app.forecasting.statistical.base.fitting import arma_innovations

        e, fitted = arma_innovations(y, np.array([]), self._theta, intercept=self._intercept)
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
        assert self._y is not None
        path = forecast_arma(
            self._y,
            self.residuals(),
            np.array([]),
            self._theta,
            intercept=self._intercept,
            horizon=h,
        )
        regime = (
            frame[self._regime_column][-1]
            if self._regime_column and self._regime_column in frame.columns
            else None
        )
        return self._build_forecast(path, horizon=h, regime_used=regime)

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "q": self._q,
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
        self._q = state.get("q")
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
