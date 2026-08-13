"""Heston stochastic volatility model."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.simulation.base.generator import (
    GeneratorMeta,
    PathGenerator,
    PathResult,
    register_generator,
)
from iqrp.app.simulation.noise.distributions import NoiseName, sample_noise
from iqrp.app.simulation.stochastic.gbm import _squeeze


@register_generator
class HestonModel(PathGenerator):
    meta = GeneratorMeta(
        name="heston",
        version="1.0.0",
        description="Heston stochastic volatility",
        family="stochastic_volatility",
        parameters={
            "drift": 0.05,
            "kappa": 2.0,
            "theta": 0.04,
            "xi": 0.3,
            "rho": -0.7,
            "v0": 0.04,
        },
    )

    def generate(
        self,
        n_steps: int,
        *,
        x0: float | np.ndarray = 100.0,
        dt: float = 0.004,
        drift: float = 0.05,
        heston_kappa: float = 2.0,
        heston_theta: float = 0.04,
        heston_xi: float = 0.3,
        heston_rho: float = -0.7,
        volatility: float = 0.2,
        noise: NoiseName = "gaussian",
        noise_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> PathResult:
        # Single-asset Heston (primary); multi-asset falls back to independent copies
        x0_arr = np.atleast_1d(np.asarray(x0, dtype=np.float64))
        n_assets = x0_arr.size
        kappa, theta, xi, rho = heston_kappa, heston_theta, heston_xi, heston_rho
        vol0 = float(np.asarray(volatility, dtype=np.float64).ravel()[0])
        mu0 = float(np.asarray(drift, dtype=np.float64).ravel()[0])
        v0 = vol0**2
        prices = np.zeros((n_steps + 1, n_assets), dtype=np.float64)
        prices[0] = x0_arr
        vol_path = np.zeros((n_steps, n_assets), dtype=np.float64)
        drift_path = np.full((n_steps, n_assets), mu0, dtype=np.float64)
        variance = np.full(n_assets, v0, dtype=np.float64)
        kwargs = noise_kwargs or {}
        for t in range(n_steps):
            z1 = sample_noise(n_assets, noise, rng=self.rng, **kwargs)
            z2 = sample_noise(n_assets, noise, rng=self.rng, **kwargs)
            z_v = z1
            z_s = rho * z1 + np.sqrt(max(0.0, 1.0 - rho**2)) * z2
            v_pos = np.maximum(variance, 1e-12)
            variance = variance + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * z_v
            variance = np.maximum(variance, 1e-12)
            vol_path[t] = np.sqrt(variance)
            rets = (mu0 - 0.5 * variance) * dt + np.sqrt(variance * dt) * z_s
            prices[t + 1] = prices[t] * np.exp(rets)
        log_rets = np.diff(np.log(np.clip(prices, 1e-12, None)), axis=0)
        return PathResult(
            prices=_squeeze(prices),
            returns=_squeeze(log_rets),
            volatility=_squeeze(vol_path),
            drift=_squeeze(drift_path),
            latent={
                "variance": _squeeze(np.vstack([np.full(n_assets, v0), np.maximum(variance, 0)]))
            },
            metadata={"model": "heston"},
        )
