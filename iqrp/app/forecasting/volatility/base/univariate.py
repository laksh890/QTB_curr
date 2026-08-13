"""Shared helpers for univariate volatility models."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.volatility.base.likelihood import estimate
from iqrp.app.forecasting.volatility.base.recursion import forecast_garch_path
from iqrp.app.forecasting.volatility.base.volatility_model import VolatilityModel


class UnivariateVolatilityModel(VolatilityModel):
    """Template for single-series conditional variance models."""

    def _demean(self, r: np.ndarray) -> np.ndarray:
        return r - float(np.mean(r))

    def _fit_mle(
        self,
        returns: np.ndarray,
        variance_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        bounds: list[tuple[float, float]],
        param_names: list[str],
    ) -> Any:
        opt = self._vol_settings.optimization
        return estimate(
            returns,
            variance_fn,
            x0,
            bounds,
            param_names=param_names,
            dist=self._dist_name(),
            dist_kwargs=self._dist_kwargs(),
            method=opt.method,
            maxiter=opt.maxiter,
            n_restarts=opt.n_restarts,
        )

    def _regime_fit(
        self,
        returns: np.ndarray,
        regimes: np.ndarray | None,
        fit_subset: Callable[[np.ndarray], tuple[dict[str, float], np.ndarray, float, float, float]],
    ) -> tuple[dict[str, float], np.ndarray, float, float, float]:
        """Fit globally; optionally store per-regime params for adaptive switching."""
        params, var, ll, aic, bic = fit_subset(returns)
        self._regime_params = {}
        if (
            regimes is not None
            and self._vol_settings.regime.enabled
            and self._vol_settings.regime.condition
        ):
            for reg in np.unique(regimes):
                mask = regimes == reg
                if int(mask.sum()) < 30:
                    continue
                try:
                    p_r, _, _, _, _ = fit_subset(returns[mask])
                    self._regime_params[reg] = p_r
                except Exception:  # noqa: BLE001
                    continue
            if self._vol_settings.regime.ensemble_weight and self._regime_params:
                # ensemble-weighted variance using last regime frequency
                weights = {k: float(np.mean(regimes == k)) for k in self._regime_params}
                # keep global variance; store weights in params metadata via finalize extras
                params = {**params, **{f"w_{k}": w for k, w in weights.items()}}
        return params, var, ll, aic, bic

    def predict(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        assert self._variance is not None
        tgt = self._target_column or self._vol_settings.columns.target
        if tgt in frame.columns and self._returns is not None:
            r = frame[tgt].to_numpy().astype(np.float64)
            if r.size == self._variance.size and np.allclose(r[: min(10, r.size)], self._returns[: min(10, r.size)]):
                return np.sqrt(self._variance)
            # recompute on new series with frozen params
            return np.sqrt(self._variance_from_returns(r))
        return np.sqrt(self._variance)

    def _variance_from_returns(self, returns: np.ndarray) -> np.ndarray:
        """Override in subclasses that support out-of-sample variance recursion."""
        assert self._variance is not None
        if returns.size == self._variance.size:
            return self._variance
        # pad / truncate fallback
        v = self._variance
        if returns.size < v.size:
            return v[-returns.size :]
        pad = np.full(returns.size - v.size, v[-1])
        return np.concatenate([v, pad])

    def _garch11_forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._returns is not None and self._variance is not None
        omega = float(self._params.get("omega", 1e-4))
        alpha = float(self._params.get("alpha", self._params.get("alpha_0", 0.05)))
        beta = float(self._params.get("beta", self._params.get("beta_0", 0.9)))
        last_eps2 = float(self._returns[-1] ** 2)
        last_var = float(self._variance[-1])
        var = forecast_garch_path(last_eps2, last_var, omega, alpha, beta, horizon=horizon)
        return np.sqrt(var), var

    def forecast(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        feature_columns: list[str] | None = None,
    ) -> Forecast:
        self._require_fitted()
        h = self._default_horizon(horizon)
        sigma, var = self._forecast_path(h)
        # scenario paths
        n_scen = int(self._vol_settings.forecast.scenario_paths)
        meta_extra: dict[str, Any] = {}
        if n_scen > 0:
            rng = np.random.default_rng(0)
            shocks = rng.normal(size=(n_scen, h))
            scen = np.sqrt(var)[None, :] * np.abs(shocks)
            meta_extra["scenarios"] = scen.tolist()
        regime_used = None
        if self._regime_column and self._regime_column in frame.columns:
            regime_used = frame[self._regime_column].to_numpy()[-1]
            if regime_used in self._regime_params and self._vol_settings.regime.condition:
                # adaptive: swap params temporarily for path
                saved = dict(self._params)
                self._params = {**saved, **self._regime_params[regime_used]}
                sigma, var = self._forecast_path(h)
                self._params = saved
        fc = self._build_vol_forecast(sigma, var, horizon=h, regime_used=regime_used)
        if meta_extra:
            fc = Forecast.from_values(
                sigma,
                horizon=h,
                model_name=self.meta.name,
                model_version=self.meta.version,
                features_used=tuple(self._feature_columns),
                regime_used=regime_used,
                strategy="recursive",
                intervals=fc.intervals,
                metadata={**fc.metadata, **meta_extra},
            )
        return fc

    def _forecast_path(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        return self._garch11_forecast(horizon)
