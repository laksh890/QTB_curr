"""RiskMetrics-style EWMA volatility."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.volatility.base.likelihood import gaussian_nll_from_variance
from iqrp.app.forecasting.volatility.base.recursion import ewma_variance
from iqrp.app.forecasting.volatility.base.univariate import UnivariateVolatilityModel


@register_forecast_model
class EWMAVolatilityModel(UnivariateVolatilityModel):
    meta = ForecastModelMeta(
        name="ewma",
        version="1.0.0",
        description="Exponentially Weighted Moving Average volatility",
        algorithm_family="volatility",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_intervals=True,
    )

    def __init__(self, settings: Any | None = None, *, lam: float | None = None, **params: Any) -> None:
        super().__init__(settings=settings, **params)
        self._lam = lam

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> EWMAVolatilityModel:
        tgt = self._resolve_target_name(frame, target_column)
        r = self._demean(frame[tgt].to_numpy().astype(np.float64))
        regimes = self._maybe_regime(frame, regime_column)

        def _fit_subset(rr: np.ndarray) -> tuple[dict[str, float], np.ndarray, float, float, float]:
            lam0 = float(self._lam if self._lam is not None else self._vol_settings.order.ewma_lambda)

            def var_fn(theta: np.ndarray) -> np.ndarray:
                return ewma_variance(rr, float(theta[0]))

            # grid + optional MLE for lambda
            if self._params_kw.get("estimate_lambda", True):
                res = self._fit_mle(
                    rr,
                    var_fn,
                    np.array([lam0]),
                    [(0.80, 0.999)],
                    ["lambda"],
                )
                lam = float(res.params[0])
                var = res.variance
                ll, aic, bic = res.loglik, res.aic, res.bic
            else:
                lam = lam0
                var = ewma_variance(rr, lam)
                nll = gaussian_nll_from_variance(rr, var, dist=self._dist_name(), dist_kwargs=self._dist_kwargs())
                ll = -nll
                aic, bic = -2 * ll + 2, -2 * ll + np.log(max(rr.size, 1))
            return {"lambda": lam}, var, ll, aic, bic

        params, var, ll, aic, bic = self._regime_fit(r, regimes, _fit_subset)
        self._lam = float(params["lambda"])
        self._finalize(r, var, target_column=tgt, params=params, loglik=ll, aic=aic, bic=bic)
        return self

    def _variance_from_returns(self, returns: np.ndarray) -> np.ndarray:
        return ewma_variance(returns, float(self._params.get("lambda", 0.94)))

    def _forecast_path(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._returns is not None and self._variance is not None
        lam = float(self._params.get("lambda", 0.94))
        last = float(self._variance[-1])
        # EWMA multi-step: variance forecast stays at last conditional var for riskmetrics
        # more precisely E[σ²_{t+h}] → last for IGARCH-like EWMA
        var = np.full(horizon, last)
        # 1-step update using last shock
        var[0] = lam * last + (1 - lam) * float(self._returns[-1] ** 2)
        if horizon > 1:
            var[1:] = var[0]
        return np.sqrt(var), var

    def _algorithm_state(self) -> dict[str, Any]:
        state = super()._algorithm_state()
        state["lam"] = self._lam
        return state

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        super()._load_algorithm_state(state)
        self._lam = state.get("lam")
