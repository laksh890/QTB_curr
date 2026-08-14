"""Component GARCH (short-run / long-run) model."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.volatility.base.recursion import cgarch_variance, forecast_garch_path
from iqrp.app.forecasting.volatility.base.univariate import UnivariateVolatilityModel


@register_forecast_model
class ComponentGARCHModel(UnivariateVolatilityModel):
    meta = ForecastModelMeta(
        name="component_garch",
        version="1.0.0",
        description="Component GARCH with permanent/transitory variance",
        algorithm_family="volatility",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_intervals=True,
    )

    def __init__(self, settings: Any | None = None, **params: Any) -> None:
        super().__init__(settings=settings, **params)
        self._q_path: np.ndarray | None = None

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> ComponentGARCHModel:
        tgt = self._resolve_target_name(frame, target_column)
        r = self._demean(frame[tgt].to_numpy().astype(np.float64))
        regimes = self._maybe_regime(frame, regime_column)

        def _fit_subset(rr: np.ndarray) -> tuple[dict[str, float], np.ndarray, float, float, float]:
            mean2 = float(np.mean(rr**2))

            def var_fn(theta: np.ndarray) -> np.ndarray:
                omega, rho, phi, alpha, beta = map(float, theta)
                h, _ = cgarch_variance(rr, omega, rho, phi, alpha, beta)
                return h

            x0 = np.array([0.05 * mean2, 0.95, 0.05, 0.05, 0.85])
            bounds = [
                (1e-12, 10 * mean2),
                (0.5, 0.999),
                (0.0, 0.5),
                (0.0, 0.5),
                (0.0, 0.99),
            ]
            names = ["omega", "rho", "phi", "alpha", "beta"]
            res = self._fit_mle(rr, var_fn, x0, bounds, names)
            params = {n: float(v) for n, v in zip(names, res.params)}
            _, q = cgarch_variance(rr, *[params[n] for n in names])
            self._q_path = q
            return params, res.variance, res.loglik, res.aic, res.bic

        params, var, ll, aic, bic = self._regime_fit(r, regimes, _fit_subset)
        self._finalize(
            r,
            var,
            target_column=tgt,
            params=params,
            loglik=ll,
            aic=aic,
            bic=bic,
            extras={
                "permanent_last": float(self._q_path[-1]) if self._q_path is not None else None
            },
        )
        return self

    def _variance_from_returns(self, returns: np.ndarray) -> np.ndarray:
        h, q = cgarch_variance(
            returns,
            float(self._params["omega"]),
            float(self._params["rho"]),
            float(self._params["phi"]),
            float(self._params["alpha"]),
            float(self._params["beta"]),
        )
        self._q_path = q
        return h

    def _forecast_path(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._returns is not None and self._variance is not None
        persist = float(self._params["alpha"]) + float(self._params["beta"])
        var = forecast_garch_path(
            float(self._returns[-1] ** 2),
            float(self._variance[-1]),
            float(self._params["omega"]),
            persist * 0.2,
            persist * 0.8 if persist else 0.8,
            horizon=horizon,
        )
        return np.sqrt(var), var

    def _algorithm_state(self) -> dict[str, Any]:
        state = super()._algorithm_state()
        state["q_path"] = None if self._q_path is None else self._q_path.tolist()
        return state

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        super()._load_algorithm_state(state)
        q = state.get("q_path")
        self._q_path = None if q is None else np.asarray(q, dtype=np.float64)
