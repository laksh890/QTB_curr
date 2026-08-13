"""EGARCH volatility model."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.volatility.base.recursion import egarch_variance
from iqrp.app.forecasting.volatility.base.univariate import UnivariateVolatilityModel


@register_forecast_model
class EGARCHModel(UnivariateVolatilityModel):
    meta = ForecastModelMeta(
        name="egarch",
        version="1.0.0",
        description="Exponential GARCH (Nelson)",
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
    ) -> EGARCHModel:
        tgt = self._resolve_target_name(frame, target_column)
        r = self._demean(frame[tgt].to_numpy().astype(np.float64))
        regimes = self._maybe_regime(frame, regime_column)

        def _fit_subset(rr: np.ndarray) -> tuple[dict[str, float], np.ndarray, float, float, float]:
            def var_fn(theta: np.ndarray) -> np.ndarray:
                omega, alpha, gamma, beta = map(float, theta)
                return egarch_variance(
                    rr,
                    omega,
                    np.array([alpha]),
                    np.array([gamma]),
                    np.array([beta]),
                )

            x0 = np.array([-0.1, 0.1, -0.05, 0.95])
            bounds = [(-5.0, 5.0), (-1.0, 1.0), (-1.0, 1.0), (-0.999, 0.999)]
            names = ["omega", "alpha", "gamma", "beta"]
            res = self._fit_mle(rr, var_fn, x0, bounds, names)
            params = {n: float(v) for n, v in zip(names, res.params)}
            return params, res.variance, res.loglik, res.aic, res.bic

        params, var, ll, aic, bic = self._regime_fit(r, regimes, _fit_subset)
        self._finalize(r, var, target_column=tgt, params=params, loglik=ll, aic=aic, bic=bic)
        return self

    def _variance_from_returns(self, returns: np.ndarray) -> np.ndarray:
        return egarch_variance(
            returns,
            float(self._params["omega"]),
            np.array([self._params["alpha"]]),
            np.array([self._params["gamma"]]),
            np.array([self._params["beta"]]),
        )

    def _forecast_path(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        # approximate multi-step via last variance persistence on log scale
        assert self._variance is not None
        beta = float(self._params.get("beta", 0.9))
        last = float(self._variance[-1])
        log_last = np.log(max(last, 1e-12))
        omega = float(self._params.get("omega", -0.1))
        uncond = np.exp(omega / max(1 - abs(beta), 1e-6))
        var = np.empty(horizon)
        var[0] = last
        for i in range(1, horizon):
            log_v = np.log(uncond) + (beta**i) * (log_last - np.log(uncond))
            var[i] = max(np.exp(log_v), 1e-12)
        return np.sqrt(var), var
