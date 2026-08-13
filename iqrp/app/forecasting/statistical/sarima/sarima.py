"""Seasonal ARIMA SARIMA(p,d,q)(P,D,Q)s forecasting model."""

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
from iqrp.app.forecasting.statistical.base.stationarity import (
    difference,
    integrate,
    seasonal_difference,
    seasonal_integrate,
    suggest_differencing,
    suggest_seasonal_differencing,
)
from iqrp.app.forecasting.statistical.base.statistical_model import StatisticalForecastModel


@register_forecast_model
class SARIMAModel(StatisticalForecastModel):
    meta = ForecastModelMeta(
        name="sarima",
        version="1.0.0",
        description="Seasonal ARIMA model",
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
        p: int | None = None,
        d: int | None = None,
        q: int | None = None,
        P: int | None = None,
        D: int | None = None,
        Q: int | None = None,
        seasonal_period: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(settings=settings, **kwargs)
        self._p, self._d, self._q = p, d, q
        self._P, self._D, self._Q = P, D, Q
        self._s = seasonal_period
        self._phi = np.array([], dtype=np.float64)
        self._theta = np.array([], dtype=np.float64)
        self._Phi = np.array([], dtype=np.float64)
        self._Theta = np.array([], dtype=np.float64)
        self._intercept = 0.0

    def _prepare_series(self, y: np.ndarray) -> np.ndarray:
        s = int(self._s or self._stat_settings.order.seasonal_period)
        D = int(self._D or 0)
        d = int(self._d or 0)
        z = y
        if D:
            z = seasonal_difference(z, period=s, order=D)
        if d:
            z = difference(z, order=d)
        return z

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> SARIMAModel:
        tgt = self._resolve_target_name(frame, target_column)
        y = frame[tgt].to_numpy().astype(np.float64)
        regimes = self._maybe_regime_series(frame, regime_column)
        y_fit = self._regime_conditioned_y(y, regimes)
        s = int(self._s or self._stat_settings.order.seasonal_period)
        if self._stat_settings.identification.auto:
            d = int(self._d if self._d is not None else suggest_differencing(y_fit, max_d=self._stat_settings.order.max_d))
            D = int(
                self._D
                if self._D is not None
                else (
                    suggest_seasonal_differencing(
                        y_fit, period=s, max_D=self._stat_settings.order.max_D
                    )
                    if self._stat_settings.identification.seasonal_detect
                    else 0
                )
            )
            p = int(self._p if self._p is not None else (self._stat_settings.order.p or 1))
            q = int(self._q if self._q is not None else (self._stat_settings.order.q or 0))
            P = int(self._P if self._P is not None else min(1, self._stat_settings.order.max_P))
            Q = int(self._Q if self._Q is not None else min(1, self._stat_settings.order.max_Q))
        else:
            d = int(self._d or 0)
            D = int(self._D or 0)
            p = int(self._p or 1)
            q = int(self._q or 0)
            P = int(self._P or 0)
            Q = int(self._Q or 0)
        self._p, self._d, self._q, self._P, self._D, self._Q, self._s = p, d, q, P, D, Q, s
        z = self._prepare_series(y_fit)
        # expand seasonal AR/MA into long ARMA lags approximately via seasonal lag inclusion
        # Fit nonseasonal ARMA on seasonally differenced series; then add seasonal AR via OLS on seasonal lags
        fit = fit_arma_css(z, p, q, intercept=True)
        self._phi = np.asarray(fit.metadata.get("ar") or [], dtype=np.float64)
        self._theta = np.asarray(fit.metadata.get("ma") or [], dtype=np.float64)
        self._intercept = fit.intercept
        # seasonal AR on residuals of nonseasonal
        if P > 0 and z.size > s * P + 5:
            from iqrp.app.forecasting.statistical.base.fitting import fit_ar_ols

            # use every s-th lag of z
            zs = z[::1]
            # build seasonal lag design
            rows = zs.size - s * P
            if rows > 5:
                Y = zs[s * P :]
                X = np.column_stack([zs[s * P - s * k : zs.size - s * k] for k in range(1, P + 1)])
                beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
                self._Phi = np.asarray(beta, dtype=np.float64)
            else:
                self._Phi = np.zeros(P)
        else:
            self._Phi = np.zeros(max(P, 0))
        self._Theta = np.zeros(max(Q, 0))
        # fitted on original scale
        e, fitted_z = arma_innovations(z, self._phi, self._theta, intercept=self._intercept)
        # apply seasonal AR correction
        if self._Phi.size and fitted_z.size > s:
            for t in range(s, fitted_z.size):
                for k in range(self._Phi.size):
                    idx = t - s * (k + 1)
                    if idx >= 0:
                        fitted_z[t] += self._Phi[k] * z[idx]
            e = z - fitted_z
        recon = fitted_z
        if d:
            recon = integrate(recon, seasonal_difference(y_fit, period=s, order=D) if D else y_fit, order=d)
        if D:
            hist_for_seas = y_fit if not d else y_fit
            recon = seasonal_integrate(recon, hist_for_seas, period=s, order=D)
        if recon.size < y.size:
            recon = np.concatenate([y[: y.size - recon.size], recon])
        resid = y - recon[-y.size :]
        ic = information_criteria(fit.loglik, fit.k_params + P + Q, fit.nobs)
        self._finalize_fit(
            y,
            target_column=tgt,
            feature_columns=feature_columns or [tgt],
            residuals=resid[-y.size :],
            fitted=recon[-y.size :],
            sigma2=float(np.var(e)) or fit.sigma2,
            order={"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "s": s},
            ic=ic,
        )
        return self

    def predict(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        # use stored fitted if same length else recompute via forecast of length 0 path
        tgt = self._target_column or self._stat_settings.columns.target
        y = frame[tgt].to_numpy().astype(np.float64) if tgt in frame.columns else self._extract_target(frame, None)
        if self._fitted_values is not None and self._fitted_values.size == y.size:
            return self._fitted_values.copy()
        # one-step recursive in-sample
        out = np.empty(y.size, dtype=np.float64)
        for t in range(y.size):
            if t == 0:
                out[t] = y[t]
            else:
                fc = self.forecast(frame.slice(0, t + 1), horizon=1)
                out[t] = float(fc.values[0])
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
        s = int(self._s or 12)
        d, D = int(self._d or 0), int(self._D or 0)
        z = self._prepare_series(self._y)
        e, _ = arma_innovations(z, self._phi, self._theta, intercept=self._intercept)
        z_path = forecast_arma(z, e, self._phi, self._theta, intercept=self._intercept, horizon=h)
        # seasonal AR recursion
        hist = list(z)
        for i in range(h):
            seas = 0.0
            for k in range(self._Phi.size):
                idx = len(hist) - s * (k + 1)
                if idx >= 0:
                    seas += self._Phi[k] * hist[idx]
            z_path[i] = z_path[i] + seas
            hist.append(z_path[i])
        path = z_path
        if d:
            base = seasonal_difference(self._y, period=s, order=D) if D else self._y
            path = integrate(path, base, order=d)
        if D:
            path = seasonal_integrate(path, self._y, period=s, order=D)
        regime = frame[self._regime_column][-1] if self._regime_column and self._regime_column in frame.columns else None
        return self._build_forecast(path, horizon=h, regime_used=regime)

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "p": self._p,
            "d": self._d,
            "q": self._q,
            "P": self._P,
            "D": self._D,
            "Q": self._Q,
            "s": self._s,
            "phi": self._phi.tolist(),
            "theta": self._theta.tolist(),
            "Phi": self._Phi.tolist(),
            "Theta": self._Theta.tolist(),
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
        for key in ("p", "d", "q", "P", "D", "Q", "s"):
            setattr(self, f"_{key}", state.get(key))
        self._phi = np.asarray(state.get("phi") or [], dtype=np.float64)
        self._theta = np.asarray(state.get("theta") or [], dtype=np.float64)
        self._Phi = np.asarray(state.get("Phi") or [], dtype=np.float64)
        self._Theta = np.asarray(state.get("Theta") or [], dtype=np.float64)
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
