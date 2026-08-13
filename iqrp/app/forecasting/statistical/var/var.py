"""Vector Autoregression VAR(p)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.statistical.base.fitting import forecast_var, fit_var_ols, information_criteria
from iqrp.app.forecasting.statistical.base.multivariate import fevd, granger_causality, impulse_response
from iqrp.app.forecasting.statistical.base.selection import select_var_lags
from iqrp.app.forecasting.statistical.base.statistical_model import StatisticalForecastModel


@register_forecast_model
class VARModel(StatisticalForecastModel):
    meta = ForecastModelMeta(
        name="var",
        version="1.0.0",
        description="Vector Autoregression VAR(p)",
        algorithm_family="statistical",
        task="multi_step",
        default_horizon=5,
        supports_online=True,
        supports_intervals=True,
    )

    def __init__(self, settings: Any | None = None, *, p: int | None = None, **kwargs: Any) -> None:
        super().__init__(settings=settings, **kwargs)
        self._p = p
        self._coefs: np.ndarray | None = None
        self._intercept = np.zeros(1)
        self._sigma = np.eye(1)
        self._Y: np.ndarray | None = None

    def _endog_matrix(self, frame: pl.DataFrame, feature_columns: list[str] | None) -> tuple[np.ndarray, list[str]]:
        names = list(feature_columns or [])
        if not names:
            endog = self._stat_settings.columns.endogenous
            if endog:
                names = list(endog)
            else:
                # numeric columns excluding timestamp
                ts = self._stat_settings.columns.timestamp
                names = [c for c in frame.columns if c != ts and frame[c].dtype.is_numeric()]
                tgt = self._stat_settings.columns.target
                if tgt in names:
                    names = [tgt] + [c for c in names if c != tgt]
                names = names[: min(len(names), 5)]
        Y = frame.select(names).to_numpy().astype(np.float64)
        return Y, names

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> VARModel:
        self._maybe_regime_series(frame, regime_column)
        Y, names = self._endog_matrix(frame, feature_columns)
        self._endog_names = names
        p = self._p
        if p is None:
            if self._stat_settings.identification.auto:
                p = int(
                    select_var_lags(
                        Y,
                        max_lags=self._stat_settings.order.max_var_lags,
                        criterion=self._stat_settings.identification.criterion,  # type: ignore[arg-type]
                    ).best_order["p"]
                )
            else:
                p = int(self._stat_settings.order.p or 1)
        fit = fit_var_ols(Y, int(p), intercept=True)
        self._p = int(p)
        self._coefs = fit["coefs"]
        self._intercept = fit["intercept"]
        self._sigma = fit["sigma"]
        self._Y = Y
        resid = fit["residuals"]
        # pad residuals
        pad = Y.shape[0] - resid.shape[0]
        if pad > 0:
            resid_full = np.vstack([np.zeros((pad, Y.shape[1])), resid])
            fitted_full = np.vstack([Y[:pad], fit["fitted"]])
        else:
            resid_full, fitted_full = resid, fit["fitted"]
        ic = information_criteria(fit["loglik"], fit["k_params"], fit["nobs"])
        tgt = target_column or names[0]
        self._finalize_fit(
            Y[:, 0],
            target_column=tgt,
            feature_columns=names,
            residuals=resid_full[:, 0],
            fitted=fitted_full[:, 0],
            sigma2=float(np.mean(np.diag(self._sigma))),
            order={"p": int(p), "k": Y.shape[1]},
            ic=ic,
            algorithm_extras={"endog": names},
        )
        self._residuals = resid_full[:, 0]
        return self

    def predict(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        Y, _ = self._endog_matrix(frame, feature_columns or self._endog_names)
        assert self._coefs is not None
        p = self._coefs.shape[0]
        out = np.empty(Y.shape[0], dtype=np.float64)
        for t in range(Y.shape[0]):
            if t < p:
                out[t] = Y[t, 0]
            else:
                yhat = self._intercept.copy()
                for lag in range(p):
                    yhat = yhat + self._coefs[lag] @ Y[t - 1 - lag]
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
        path_m = forecast_var(self._Y, self._coefs, self._intercept, horizon=h)
        path = path_m[:, 0]
        regime = frame[self._regime_column][-1] if self._regime_column and self._regime_column in frame.columns else None
        fc = self._build_forecast(path, horizon=h, regime_used=regime)
        fc.metadata["multivariate_path"] = path_m.tolist()
        return fc

    def impulse_response(self, *, horizon: int = 10, orthogonal: bool = True) -> np.ndarray:
        self._require_fitted()
        assert self._coefs is not None
        return impulse_response(self._coefs, self._sigma, horizon=horizon, orthogonal=orthogonal)

    def fevd(self, *, horizon: int = 10) -> np.ndarray:
        self._require_fitted()
        assert self._coefs is not None
        return fevd(self._coefs, self._sigma, horizon=horizon)

    def granger(self, cause: int, effect: int, *, lag: int | None = None) -> Any:
        self._require_fitted()
        assert self._Y is not None
        return granger_causality(self._Y, cause=cause, effect=effect, lag=lag or int(self._p or 1))

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "p": self._p,
            "coefs": None if self._coefs is None else self._coefs.tolist(),
            "intercept": self._intercept.tolist(),
            "sigma": self._sigma.tolist(),
            "Y": None if self._Y is None else self._Y.tolist(),
            "endog_names": self._endog_names,
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
        self._coefs = None if state.get("coefs") is None else np.asarray(state["coefs"], dtype=np.float64)
        self._intercept = np.asarray(state.get("intercept") or [0.0], dtype=np.float64)
        self._sigma = np.asarray(state.get("sigma") or [[1.0]], dtype=np.float64)
        self._Y = None if state.get("Y") is None else np.asarray(state["Y"], dtype=np.float64)
        self._endog_names = list(state.get("endog_names") or [])
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
        self._y = None if state.get("y") is None else np.asarray(state["y"], dtype=np.float64)
