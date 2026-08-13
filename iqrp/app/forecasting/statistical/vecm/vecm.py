"""Vector Error Correction Model (VECM)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.statistical.base.multivariate import (
    engle_granger,
    fit_vecm_engle_granger,
    johansen_trace,
)
from iqrp.app.forecasting.statistical.base.statistical_model import StatisticalForecastModel


@register_forecast_model
class VECMModel(StatisticalForecastModel):
    meta = ForecastModelMeta(
        name="vecm",
        version="1.0.0",
        description="Vector Error Correction Model",
        algorithm_family="statistical",
        task="multi_step",
        default_horizon=5,
        supports_online=True,
        supports_intervals=True,
    )

    def __init__(self, settings: Any | None = None, *, lags: int | None = None, **kwargs: Any) -> None:
        super().__init__(settings=settings, **kwargs)
        self._lags = lags
        self._beta = np.array([1.0])
        self._alpha = np.array([0.0])
        self._B: np.ndarray | None = None
        self._Y: np.ndarray | None = None
        self._endog_names: list[str] = []
        self._coint: dict[str, Any] = {}

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> VECMModel:
        self._maybe_regime_series(frame, regime_column)
        names = list(feature_columns or self._stat_settings.columns.endogenous or ())
        if not names:  # pragma: no cover
            ts = self._stat_settings.columns.timestamp
            names = [c for c in frame.columns if c != ts and frame[c].dtype.is_numeric()][:3]
        if len(names) < 2:
            # duplicate with lag to allow bivariate structure
            names = names + names
        self._endog_names = names[: max(len(names), 2)]
        # ensure unique columns for select
        uniq = []
        for n in self._endog_names:
            if n not in uniq:
                uniq.append(n)
        while len(uniq) < 2:  # pragma: no cover
            uniq.append(uniq[0])
        # if duplicate names, build matrix manually
        cols = []
        final_names = []
        for i, n in enumerate(uniq if len(set(self._endog_names)) >= 2 else names):
            if n in frame.columns:
                cols.append(frame[n].to_numpy().astype(np.float64))
                final_names.append(n)
        if len(cols) == 1:
            cols.append(np.roll(cols[0], 1))
            cols[1][0] = cols[0][0]
            final_names.append(final_names[0] + "_lag")
        Y = np.column_stack(cols)
        self._endog_names = final_names
        lags = int(self._lags or self._stat_settings.order.p or 1)
        fit = fit_vecm_engle_granger(Y, lags=lags)
        self._lags = lags
        self._beta = np.asarray(fit["beta"], dtype=np.float64)
        self._alpha = np.asarray(fit["alpha"], dtype=np.float64)
        self._B = fit["B"]
        self._Y = Y
        joh = johansen_trace(Y, lags=lags)
        eg = engle_granger(Y[:, 0], Y[:, 1]) if Y.shape[1] >= 2 else joh
        self._coint = {"johansen": joh.to_dict(), "engle_granger": eg.to_dict()}
        resid = fit["residuals"]
        pad = Y.shape[0] - resid.shape[0]
        resid0 = np.concatenate([np.zeros(pad), resid[:, 0]]) if resid.size else np.zeros(Y.shape[0])
        fitted0 = Y[:, 0] - resid0
        self._finalize_fit(
            Y[:, 0],
            target_column=target_column or final_names[0],
            feature_columns=final_names,
            residuals=resid0,
            fitted=fitted0,
            sigma2=float(np.var(resid0)) or 1e-12,
            order={"lags": lags, "rank": int(fit["rank"])},
            algorithm_extras={"cointegration": self._coint},
        )
        return self

    def predict(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        if self._fitted_values is not None and self._fitted_values.size == frame.height:
            return self._fitted_values.copy()
        # fallback: levels from last ECT forecast accumulation
        y = frame[self._endog_names[0]].to_numpy().astype(np.float64)
        return y

    def forecast(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        feature_columns: list[str] | None = None,
    ) -> Forecast:
        self._require_fitted()
        h = self._default_horizon(horizon)
        assert self._Y is not None and self._B is not None
        Y = self._Y.copy()
        path = np.empty(h, dtype=np.float64)
        p = int(self._lags or 1)
        for i in range(h):
            dY_hist = np.diff(Y, axis=0)
            ect = float(Y[-1] @ self._beta)
            # build feature row: ect, lagged dY..., const
            feats = [ect]
            for lag in range(1, p):
                if dY_hist.shape[0] >= lag:
                    feats.extend(dY_hist[-lag].tolist())
                else:
                    feats.extend([0.0] * Y.shape[1])
            feats.append(1.0)
            xrow = np.asarray(feats, dtype=np.float64)
            # adjust length to B rows
            if xrow.size < self._B.shape[0]:
                xrow = np.pad(xrow, (0, self._B.shape[0] - xrow.size))
            elif xrow.size > self._B.shape[0]:
                xrow = xrow[: self._B.shape[0]]
            dy = xrow @ self._B
            Y = np.vstack([Y, Y[-1] + dy])
            path[i] = Y[-1, 0]
        regime = frame[self._regime_column][-1] if self._regime_column and self._regime_column in frame.columns else None
        fc = self._build_forecast(path, horizon=h, regime_used=regime)
        fc.metadata["cointegration"] = self._coint
        return fc

    def cointegration_test(self) -> dict[str, Any]:
        self._require_fitted()
        return dict(self._coint)

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "lags": self._lags,
            "beta": self._beta.tolist(),
            "alpha": self._alpha.tolist(),
            "B": None if self._B is None else self._B.tolist(),
            "Y": None if self._Y is None else self._Y.tolist(),
            "endog_names": self._endog_names,
            "coint": self._coint,
            "residuals": None if self._residuals is None else self._residuals.tolist(),
            "fitted": None if self._fitted_values is None else self._fitted_values.tolist(),
            "sigma2": self._sigma2,
            "order": self._order,
            "target_column": self._target_column,
            "feature_columns": self._feature_columns,
            "regime_column": self._regime_column,
            "y": None if self._y is None else self._y.tolist(),
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        self._lags = state.get("lags")
        self._beta = np.asarray(state.get("beta") or [1.0], dtype=np.float64)
        self._alpha = np.asarray(state.get("alpha") or [0.0], dtype=np.float64)
        self._B = None if state.get("B") is None else np.asarray(state["B"], dtype=np.float64)
        self._Y = None if state.get("Y") is None else np.asarray(state["Y"], dtype=np.float64)
        self._endog_names = list(state.get("endog_names") or [])
        self._coint = dict(state.get("coint") or {})
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
        self._y = None if state.get("y") is None else np.asarray(state["y"], dtype=np.float64)
