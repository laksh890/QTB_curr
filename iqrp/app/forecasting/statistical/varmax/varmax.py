"""VARMAX: VAR with exogenous regressors and optional MA residuals."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.statistical.base.fitting import forecast_var, information_criteria
from iqrp.app.forecasting.statistical.base.statistical_model import StatisticalForecastModel


@register_forecast_model
class VARMAXModel(StatisticalForecastModel):
    meta = ForecastModelMeta(
        name="varmax",
        version="1.0.0",
        description="VARMAX with exogenous variables",
        algorithm_family="statistical",
        task="multi_step",
        default_horizon=5,
        supports_online=True,
        supports_intervals=True,
    )

    def __init__(
        self,
        settings: Any | None = None,
        *,
        p: int | None = None,
        q: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(settings=settings, **kwargs)
        self._p = p
        self._q = q if q is not None else 0
        self._coefs: np.ndarray | None = None
        self._exog_coef: np.ndarray | None = None
        self._ma_coef: np.ndarray | None = None
        self._intercept = np.zeros(1)
        self._sigma = np.eye(1)
        self._Y: np.ndarray | None = None
        self._X: np.ndarray | None = None
        self._endog_names: list[str] = []
        self._exog_names: list[str] = []

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> VARMAXModel:
        self._maybe_regime_series(frame, regime_column)
        # split endogenous / exogenous
        endog = list(self._stat_settings.columns.endogenous or ())
        exog = list(self._stat_settings.columns.exogenous or ())
        if not endog:
            # use first two numeric as endog, rest as exog if provided via feature_columns
            cols = feature_columns or [
                c
                for c in frame.columns
                if c not in {self._stat_settings.columns.timestamp} and frame[c].dtype.is_numeric()
            ]
            endog = cols[:2] if len(cols) >= 2 else cols[:1]
            exog = [c for c in cols if c not in endog]
        self._endog_names = endog
        self._exog_names = exog
        Y = frame.select(endog).to_numpy().astype(np.float64)
        X = frame.select(exog).to_numpy().astype(np.float64) if exog else np.zeros((Y.shape[0], 0))
        p = int(self._p or self._stat_settings.order.p or 1)
        q = int(self._q or 0)
        T, K = Y.shape
        # design: intercept + lags + exog
        rows = T - p
        target = Y[p:]
        cols = [np.ones((rows, 1))]
        for lag in range(1, p + 1):
            cols.append(Y[p - lag : T - lag])
        if X.shape[1]:
            cols.append(X[p:])
        design = np.concatenate(cols, axis=1)
        B, *_ = np.linalg.lstsq(design, target, rcond=None)
        fitted = design @ B
        resid = target - fitted
        # optional MA(q) on residuals (diagonal)
        ma = np.zeros((max(q, 0), K), dtype=np.float64)
        if q > 0 and resid.shape[0] > q + 2:
            for j in range(K):
                # AR on residuals as MA proxy via OLS of e_t on e_{t-1..q}
                e = resid[:, j]
                Ye = e[q:]
                Xe = np.column_stack([e[q - i : e.size - i] for i in range(1, q + 1)])
                bj, *_ = np.linalg.lstsq(Xe, Ye, rcond=None)
                ma[:, j] = bj
                # adjust fitted
                for t in range(q, resid.shape[0]):
                    fitted[t, j] += float(bj @ resid[t - q : t, j][::-1])
            resid = target - fitted
        # unpack VAR coefs
        coefs = np.zeros((p, K, K), dtype=np.float64)
        for lag in range(p):
            block = B[1 + lag * K : 1 + (lag + 1) * K, :]
            coefs[lag] = block.T
        offset = 1 + p * K
        exog_coef = B[offset:, :].T if X.shape[1] else np.zeros((K, 0))
        sigma = (resid.T @ resid) / max(resid.shape[0] - design.shape[1], 1)
        self._p, self._q = p, q
        self._coefs = coefs
        self._exog_coef = exog_coef
        self._ma_coef = ma
        self._intercept = B[0, :]
        self._sigma = np.atleast_2d(sigma)
        self._Y, self._X = Y, X
        pad = p
        resid_full = np.vstack([np.zeros((pad, K)), resid])
        fitted_full = np.vstack([Y[:pad], fitted])
        # loglik approx
        sign, logdet = np.linalg.slogdet(self._sigma + 1e-12 * np.eye(K))
        ll = -0.5 * resid.shape[0] * (K * np.log(2 * np.pi) + (logdet if sign > 0 else 0.0))
        ic = information_criteria(ll, int(B.size), int(resid.shape[0]))
        self._finalize_fit(
            Y[:, 0],
            target_column=target_column or endog[0],
            feature_columns=endog + exog,
            residuals=resid_full[:, 0],
            fitted=fitted_full[:, 0],
            sigma2=float(np.mean(np.diag(self._sigma))),
            order={"p": p, "q": q, "k": K},
            ic=ic,
        )
        return self

    def predict(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
        self._require_fitted()
        # delegate to one-step using stored coefs
        Y = frame.select(self._endog_names).to_numpy().astype(np.float64)
        X = (
            frame.select(self._exog_names).to_numpy().astype(np.float64)
            if self._exog_names
            else np.zeros((Y.shape[0], 0))
        )
        assert self._coefs is not None
        p = self._coefs.shape[0]
        out = np.empty(Y.shape[0])
        for t in range(Y.shape[0]):
            if t < p:
                out[t] = Y[t, 0]
                continue
            yhat = self._intercept.copy()
            for lag in range(p):
                yhat = yhat + self._coefs[lag] @ Y[t - 1 - lag]
            if self._exog_coef is not None and self._exog_coef.size and t < X.shape[0]:
                yhat = yhat + self._exog_coef @ X[t]
            out[t] = yhat[0]
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
        assert self._Y is not None and self._coefs is not None
        # freeze last exog
        path_m = forecast_var(self._Y, self._coefs, self._intercept, horizon=h)
        if (
            self._exog_coef is not None
            and self._exog_coef.size
            and self._X is not None
            and self._X.size
        ):
            x_last = self._X[-1]
            for i in range(h):
                path_m[i] = path_m[i] + self._exog_coef @ x_last
        path = path_m[:, 0]
        regime = (
            frame[self._regime_column][-1]
            if self._regime_column and self._regime_column in frame.columns
            else None
        )
        return self._build_forecast(path, horizon=h, regime_used=regime)

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "p": self._p,
            "q": self._q,
            "coefs": None if self._coefs is None else self._coefs.tolist(),
            "exog_coef": None if self._exog_coef is None else self._exog_coef.tolist(),
            "ma_coef": None if self._ma_coef is None else self._ma_coef.tolist(),
            "intercept": self._intercept.tolist(),
            "sigma": self._sigma.tolist(),
            "Y": None if self._Y is None else self._Y.tolist(),
            "X": None if self._X is None else self._X.tolist(),
            "endog_names": self._endog_names,
            "exog_names": self._exog_names,
            "residuals": None if self._residuals is None else self._residuals.tolist(),
            "fitted": None if self._fitted_values is None else self._fitted_values.tolist(),
            "sigma2": self._sigma2,
            "order": self._order,
            "ic": self._ic,
            "target_column": self._target_column,
            "feature_columns": self._feature_columns,
            "regime_column": self._regime_column,
            "y": None if self._y is None else self._y.tolist(),
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        self._p = state.get("p")
        self._q = state.get("q")
        self._coefs = (
            None if state.get("coefs") is None else np.asarray(state["coefs"], dtype=np.float64)
        )
        self._exog_coef = (
            None
            if state.get("exog_coef") is None
            else np.asarray(state["exog_coef"], dtype=np.float64)
        )
        self._ma_coef = (
            None if state.get("ma_coef") is None else np.asarray(state["ma_coef"], dtype=np.float64)
        )
        self._intercept = np.asarray(state.get("intercept") or [0.0], dtype=np.float64)
        self._sigma = np.asarray(state.get("sigma") or [[1.0]], dtype=np.float64)
        self._Y = None if state.get("Y") is None else np.asarray(state["Y"], dtype=np.float64)
        self._X = None if state.get("X") is None else np.asarray(state["X"], dtype=np.float64)
        self._endog_names = list(state.get("endog_names") or [])
        self._exog_names = list(state.get("exog_names") or [])
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
        self._y = None if state.get("y") is None else np.asarray(state["y"], dtype=np.float64)
