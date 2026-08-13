"""Autoregressive AR(p) forecasting model."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.statistical.base.fitting import fit_ar_ols, forecast_arma, information_criteria
from iqrp.app.forecasting.statistical.base.selection import select_ar_order
from iqrp.app.forecasting.statistical.base.statistical_model import StatisticalForecastModel


@register_forecast_model
class ARModel(StatisticalForecastModel):
    meta = ForecastModelMeta(
        name="ar",
        version="1.0.0",
        description="Autoregressive AR(p) model",
        algorithm_family="statistical",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_intervals=True,
        parameters={},
    )

    def __init__(self, settings: Any | None = None, *, p: int | None = None, **kwargs: Any) -> None:
        super().__init__(settings=settings, **kwargs)
        self._p = p
        self._phi = np.array([], dtype=np.float64)
        self._intercept = 0.0

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> ARModel:
        tgt = self._resolve_target_name(frame, target_column)
        y = frame[tgt].to_numpy().astype(np.float64)
        regimes = self._maybe_regime_series(frame, regime_column)
        y_fit = self._regime_conditioned_y(y, regimes)
        p = self._p
        if p is None:
            if self._stat_settings.order.p is not None:
                p = int(self._stat_settings.order.p)
            elif self._stat_settings.identification.auto:
                p = int(
                    select_ar_order(
                        y_fit,
                        max_p=self._stat_settings.order.max_p,
                        criterion=self._stat_settings.identification.criterion,  # type: ignore[arg-type]
                    ).best_order["p"]
                )
            else:
                p = 1
        fit = fit_ar_ols(y_fit, int(p), intercept=True)
        self._p = int(p)
        self._phi = fit.params
        self._intercept = fit.intercept
        # align residuals to full series length (pad leading)
        resid = np.concatenate([np.zeros(max(y.size - fit.residuals.size, 0)), fit.residuals])
        fitted = np.concatenate([y[: max(y.size - fit.fitted.size, 0)], fit.fitted])
        ic = information_criteria(fit.loglik, fit.k_params, fit.nobs)
        self._finalize_fit(
            y,
            target_column=tgt,
            feature_columns=feature_columns or [tgt],
            residuals=resid[-y.size :],
            fitted=fitted[-y.size :],
            sigma2=fit.sigma2,
            order={"p": int(p)},
            ic=ic,
            algorithm_extras={"phi": self._phi.tolist(), "intercept": self._intercept},
        )
        return self

    def predict(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        tgt = self._target_column or self._stat_settings.columns.target
        y = frame[tgt].to_numpy().astype(np.float64) if tgt in frame.columns else self._extract_target(frame, None)
        p = self._phi.size
        out = np.empty(y.size, dtype=np.float64)
        for t in range(y.size):
            if t < p:
                out[t] = y[t]
            else:
                out[t] = self._intercept + float(self._phi @ y[t - p : t][::-1])
        return out

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
            self._phi,
            np.array([]),
            intercept=self._intercept,
            horizon=h,
        )
        regime = None
        if self._regime_column and self._regime_column in frame.columns:
            regime = frame[self._regime_column][-1]
        return self._build_forecast(path, horizon=h, regime_used=regime)

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "p": self._p,
            "phi": self._phi.tolist(),
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
        self._phi = np.asarray(state.get("phi") or [], dtype=np.float64)
        self._intercept = float(state.get("intercept", 0.0))
        self._y = None if state.get("y") is None else np.asarray(state["y"], dtype=np.float64)
        self._residuals = (
            None if state.get("residuals") is None else np.asarray(state["residuals"], dtype=np.float64)
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
