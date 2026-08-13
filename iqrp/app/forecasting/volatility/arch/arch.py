"""ARCH(p) volatility model."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.volatility.base.recursion import arch_variance, forecast_garch_path
from iqrp.app.forecasting.volatility.base.univariate import UnivariateVolatilityModel


@register_forecast_model
class ARCHModel(UnivariateVolatilityModel):
    meta = ForecastModelMeta(
        name="arch",
        version="1.0.0",
        description="Autoregressive Conditional Heteroskedasticity ARCH(p)",
        algorithm_family="volatility",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_intervals=True,
    )

    def __init__(self, settings: Any | None = None, *, p: int | None = None, **params: Any) -> None:
        super().__init__(settings=settings, **params)
        self._p = p

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> ARCHModel:
        tgt = self._resolve_target_name(frame, target_column)
        r = self._demean(frame[tgt].to_numpy().astype(np.float64))
        regimes = self._maybe_regime(frame, regime_column)
        p = int(self._p or self._vol_settings.order.p or 1)
        p = max(p, 1)

        def _fit_subset(rr: np.ndarray) -> tuple[dict[str, float], np.ndarray, float, float, float]:
            mean2 = float(np.mean(rr**2))

            def var_fn(theta: np.ndarray) -> np.ndarray:
                omega = float(theta[0])
                alpha = np.asarray(theta[1:], dtype=np.float64)
                return arch_variance(rr, omega, alpha)

            x0 = np.array([0.05 * mean2] + [0.1 / p] * p)
            bounds = [(1e-12, 10 * mean2)] + [(0.0, 1.0)] * p
            names = ["omega"] + [f"alpha_{i}" for i in range(p)]
            res = self._fit_mle(rr, var_fn, x0, bounds, names)
            params = {n: float(v) for n, v in zip(names, res.params)}
            return params, res.variance, res.loglik, res.aic, res.bic

        params, var, ll, aic, bic = self._regime_fit(r, regimes, _fit_subset)
        self._p = p
        self._finalize(r, var, target_column=tgt, params=params, loglik=ll, aic=aic, bic=bic, extras={"p": p})
        return self

    def _variance_from_returns(self, returns: np.ndarray) -> np.ndarray:
        p = int(self._p or 1)
        omega = float(self._params["omega"])
        alpha = np.array([self._params.get(f"alpha_{i}", 0.0) for i in range(p)], dtype=np.float64)
        return arch_variance(returns, omega, alpha)

    def _forecast_path(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._returns is not None and self._variance is not None
        omega = float(self._params["omega"])
        alpha = float(self._params.get("alpha_0", sum(v for k, v in self._params.items() if k.startswith("alpha_"))))
        var = forecast_garch_path(
            float(self._returns[-1] ** 2),
            float(self._variance[-1]),
            omega,
            alpha,
            0.0,
            horizon=horizon,
        )
        return np.sqrt(var), var

    def _algorithm_state(self) -> dict[str, Any]:
        state = super()._algorithm_state()
        state["p"] = self._p
        return state

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        super()._load_algorithm_state(state)
        self._p = state.get("p")
