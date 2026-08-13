"""FIGARCH long-memory volatility model."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.volatility.base.recursion import figarch_variance, forecast_garch_path
from iqrp.app.forecasting.volatility.base.univariate import UnivariateVolatilityModel


@register_forecast_model
class FIGARCHModel(UnivariateVolatilityModel):
    meta = ForecastModelMeta(
        name="figarch",
        version="1.0.0",
        description="Fractionally Integrated GARCH",
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
    ) -> FIGARCHModel:
        tgt = self._resolve_target_name(frame, target_column)
        r = self._demean(frame[tgt].to_numpy().astype(np.float64))
        regimes = self._maybe_regime(frame, regime_column)

        def _fit_subset(rr: np.ndarray) -> tuple[dict[str, float], np.ndarray, float, float, float]:
            mean2 = float(np.mean(rr**2))

            def var_fn(theta: np.ndarray) -> np.ndarray:
                omega, phi, d, beta = map(float, theta)
                return figarch_variance(rr, omega, phi, d, beta)

            x0 = np.array([0.05 * mean2, 0.2, 0.4, 0.4])
            bounds = [(1e-12, 10 * mean2), (0.0, 0.99), (0.01, 0.99), (0.0, 0.99)]
            names = ["omega", "phi", "d", "beta"]
            res = self._fit_mle(rr, var_fn, x0, bounds, names)
            params = {n: float(v) for n, v in zip(names, res.params)}
            return params, res.variance, res.loglik, res.aic, res.bic

        params, var, ll, aic, bic = self._regime_fit(r, regimes, _fit_subset)
        self._finalize(r, var, target_column=tgt, params=params, loglik=ll, aic=aic, bic=bic)
        return self

    def _variance_from_returns(self, returns: np.ndarray) -> np.ndarray:
        return figarch_variance(
            returns,
            float(self._params["omega"]),
            float(self._params["phi"]),
            float(self._params["d"]),
            float(self._params["beta"]),
        )

    def _forecast_path(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._returns is not None and self._variance is not None
        # hyperbolic decay approx via high persistence
        d = float(self._params.get("d", 0.4))
        beta = float(self._params.get("beta", 0.4))
        persist = min(0.99, beta + d)
        var = forecast_garch_path(
            float(self._returns[-1] ** 2),
            float(self._variance[-1]),
            float(self._params["omega"]),
            persist * 0.1,
            persist * 0.9,
            horizon=horizon,
        )
        return np.sqrt(var), var
