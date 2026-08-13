"""GJR-GARCH (threshold) volatility model."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.volatility.base.recursion import forecast_garch_path, gjr_variance
from iqrp.app.forecasting.volatility.base.univariate import UnivariateVolatilityModel


@register_forecast_model
class GJRGARCHModel(UnivariateVolatilityModel):
    meta = ForecastModelMeta(
        name="gjr_garch",
        version="1.0.0",
        description="Glosten-Jagannathan-Runkle GARCH with leverage",
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
    ) -> GJRGARCHModel:
        tgt = self._resolve_target_name(frame, target_column)
        r = self._demean(frame[tgt].to_numpy().astype(np.float64))
        regimes = self._maybe_regime(frame, regime_column)

        def _fit_subset(rr: np.ndarray) -> tuple[dict[str, float], np.ndarray, float, float, float]:
            mean2 = float(np.mean(rr**2))

            def var_fn(theta: np.ndarray) -> np.ndarray:
                omega, alpha, gamma, beta = map(float, theta)
                if alpha + 0.5 * gamma + beta >= 0.999:
                    s = 0.99 / (alpha + 0.5 * gamma + beta + 1e-12)
                    alpha, gamma, beta = alpha * s, gamma * s, beta * s
                return gjr_variance(
                    rr,
                    omega,
                    np.array([alpha]),
                    np.array([gamma]),
                    np.array([beta]),
                )

            x0 = np.array([0.05 * mean2, 0.05, 0.1, 0.85])
            bounds = [(1e-12, 10 * mean2), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
            names = ["omega", "alpha", "gamma", "beta"]
            res = self._fit_mle(rr, var_fn, x0, bounds, names)
            params = {n: float(v) for n, v in zip(names, res.params)}
            return params, res.variance, res.loglik, res.aic, res.bic

        params, var, ll, aic, bic = self._regime_fit(r, regimes, _fit_subset)
        self._finalize(r, var, target_column=tgt, params=params, loglik=ll, aic=aic, bic=bic)
        return self

    def _variance_from_returns(self, returns: np.ndarray) -> np.ndarray:
        return gjr_variance(
            returns,
            float(self._params["omega"]),
            np.array([self._params["alpha"]]),
            np.array([self._params["gamma"]]),
            np.array([self._params["beta"]]),
        )

    def _forecast_path(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._returns is not None and self._variance is not None
        # expected leverage ≈ 0.5 for symmetric shocks
        alpha_eff = float(self._params["alpha"]) + 0.5 * float(self._params["gamma"])
        var = forecast_garch_path(
            float(self._returns[-1] ** 2),
            float(self._variance[-1]),
            float(self._params["omega"]),
            alpha_eff,
            float(self._params["beta"]),
            horizon=horizon,
        )
        return np.sqrt(var), var
