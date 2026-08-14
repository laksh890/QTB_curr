"""GARCH(p,q) volatility model."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.volatility.base.recursion import forecast_garch_path, garch_variance
from iqrp.app.forecasting.volatility.base.univariate import UnivariateVolatilityModel


@register_forecast_model
class GARCHModel(UnivariateVolatilityModel):
    meta = ForecastModelMeta(
        name="garch",
        version="1.0.0",
        description="Generalized ARCH GARCH(p,q)",
        algorithm_family="volatility",
        task="regression",
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
        **params: Any,
    ) -> None:
        super().__init__(settings=settings, **params)
        self._p = p
        self._q = q

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> GARCHModel:
        tgt = self._resolve_target_name(frame, target_column)
        r = self._demean(frame[tgt].to_numpy().astype(np.float64))
        regimes = self._maybe_regime(frame, regime_column)
        p = max(int(self._p if self._p is not None else self._vol_settings.order.p), 1)
        q = max(int(self._q if self._q is not None else self._vol_settings.order.q), 1)

        def _fit_subset(rr: np.ndarray) -> tuple[dict[str, float], np.ndarray, float, float, float]:
            mean2 = float(np.mean(rr**2))

            def var_fn(theta: np.ndarray) -> np.ndarray:
                omega = float(theta[0])
                alpha = np.asarray(theta[1 : 1 + p], dtype=np.float64)
                beta = np.asarray(theta[1 + p :], dtype=np.float64)
                # soft persistence constraint via clip in recursion positivity
                if float(np.sum(alpha) + np.sum(beta)) >= 0.999:
                    scale = 0.99 / (float(np.sum(alpha) + np.sum(beta)) + 1e-12)
                    alpha = alpha * scale
                    beta = beta * scale
                return garch_variance(rr, omega, alpha, beta)

            x0 = np.array([0.05 * mean2] + [0.05 / p] * p + [0.9 / q] * q)
            bounds = [(1e-12, 10 * mean2)] + [(0.0, 1.0)] * (p + q)
            names = ["omega"] + [f"alpha_{i}" for i in range(p)] + [f"beta_{j}" for j in range(q)]
            res = self._fit_mle(rr, var_fn, x0, bounds, names)
            params = {n: float(v) for n, v in zip(names, res.params)}
            # convenience aliases for GARCH(1,1)
            if p == 1:
                params["alpha"] = params["alpha_0"]
            if q == 1:
                params["beta"] = params["beta_0"]
            return params, res.variance, res.loglik, res.aic, res.bic

        params, var, ll, aic, bic = self._regime_fit(r, regimes, _fit_subset)
        self._p, self._q = p, q
        self._finalize(
            r,
            var,
            target_column=tgt,
            params=params,
            loglik=ll,
            aic=aic,
            bic=bic,
            extras={"p": p, "q": q},
        )
        return self

    def _variance_from_returns(self, returns: np.ndarray) -> np.ndarray:
        p = int(self._p or 1)
        q = int(self._q or 1)
        omega = float(self._params["omega"])
        alpha = np.array([self._params.get(f"alpha_{i}", 0.0) for i in range(p)])
        beta = np.array([self._params.get(f"beta_{j}", 0.0) for j in range(q)])
        return garch_variance(returns, omega, alpha, beta)

    def _forecast_path(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._returns is not None and self._variance is not None
        omega = float(self._params["omega"])
        alpha = float(self._params.get("alpha", self._params.get("alpha_0", 0.05)))
        beta = float(self._params.get("beta", self._params.get("beta_0", 0.9)))
        var = forecast_garch_path(
            float(self._returns[-1] ** 2),
            float(self._variance[-1]),
            omega,
            alpha,
            beta,
            horizon=horizon,
        )
        return np.sqrt(var), var

    def _algorithm_state(self) -> dict[str, Any]:
        state = super()._algorithm_state()
        state["p"] = self._p
        state["q"] = self._q
        return state

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        super()._load_algorithm_state(state)
        self._p = state.get("p")
        self._q = state.get("q")
