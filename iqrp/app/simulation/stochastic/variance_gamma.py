"""Variance Gamma process."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.simulation.base.generator import (
    GeneratorMeta,
    PathGenerator,
    PathResult,
    register_generator,
)
from iqrp.app.simulation.stochastic.gbm import _squeeze


@register_generator
class VarianceGamma(PathGenerator):
    """Madan-Seneta Variance Gamma via time-changed Brownian motion."""

    meta = GeneratorMeta(
        name="variance_gamma",
        version="1.0.0",
        description="Variance Gamma Lévy process",
        family="levy",
        parameters={"theta": -0.1, "sigma": 0.2, "nu": 0.2, "drift": 0.05},
    )

    def generate(
        self,
        n_steps: int,
        *,
        x0: float | np.ndarray = 100.0,
        dt: float = 0.004,
        drift: float = 0.05,
        vg_theta: float = -0.1,
        vg_sigma: float = 0.2,
        vg_nu: float = 0.2,
        **_: Any,
    ) -> PathResult:
        x0_arr = np.atleast_1d(np.asarray(x0, dtype=np.float64))
        n_assets = x0_arr.size
        theta, sigma, nu = float(vg_theta), float(vg_sigma), max(float(vg_nu), 1e-8)
        mu0 = float(np.asarray(drift, dtype=np.float64).ravel()[0])
        # Gamma increments for business time: g ~ Gamma(dt/nu, nu)
        shape = dt / nu
        g = self.rng.gamma(shape, nu, size=(n_steps, n_assets))
        z = self.rng.standard_normal((n_steps, n_assets))
        # omega compensator so E[S]=exp(mu t)
        omega = np.log(1.0 - theta * nu - 0.5 * sigma**2 * nu) / nu
        increments = (mu0 + omega) * dt + theta * g + sigma * np.sqrt(g) * z
        log_prices = np.log(np.clip(x0_arr, 1e-12, None)) + np.cumsum(increments, axis=0)
        prices = np.vstack([x0_arr, np.exp(log_prices)])
        # Local vol proxy
        vol = np.full((n_steps, n_assets), sigma, dtype=np.float64)
        drift_path = np.full((n_steps, n_assets), mu0, dtype=np.float64)
        return PathResult(
            prices=_squeeze(prices),
            returns=_squeeze(increments),
            volatility=_squeeze(vol),
            drift=_squeeze(drift_path),
            latent={"gamma_time": _squeeze(g)},
            metadata={"model": "variance_gamma"},
        )
