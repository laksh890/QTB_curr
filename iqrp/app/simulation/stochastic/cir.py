"""Cox-Ingersoll-Ross square-root diffusion (for rates / variance)."""

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
class CoxIngersollRoss(PathGenerator):
    """CIR process - often used for positive rates or variance factors."""

    meta = GeneratorMeta(
        name="cir",
        version="1.0.0",
        description="Cox-Ingersoll-Ross square-root diffusion",
        family="mean_reversion",
        parameters={"kappa": 1.5, "theta": 0.04, "sigma": 0.1},
    )

    def generate(
        self,
        n_steps: int,
        *,
        x0: float | np.ndarray = 0.04,
        dt: float = 0.004,
        cir_kappa: float = 1.5,
        cir_theta: float = 0.04,
        cir_sigma: float = 0.1,
        noise: NoiseName = "gaussian",
        noise_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> PathResult:
        x0_arr = np.atleast_1d(np.asarray(x0, dtype=np.float64))
        n_assets = x0_arr.size
        kappa, theta, sigma = float(cir_kappa), float(cir_theta), float(cir_sigma)
        levels = np.zeros((n_steps + 1, n_assets), dtype=np.float64)
        levels[0] = np.maximum(x0_arr, 1e-12)
        drift_path = np.zeros((n_steps, n_assets), dtype=np.float64)
        vol_path = np.zeros((n_steps, n_assets), dtype=np.float64)
        kwargs = noise_kwargs or {}
        for t in range(n_steps):
            x_pos = np.maximum(levels[t], 1e-12)
            drift_path[t] = kappa * (theta - x_pos)
            vol_path[t] = sigma * np.sqrt(x_pos)
            z = sample_noise(n_assets, noise, rng=self.rng, **kwargs)
            levels[t + 1] = x_pos + drift_path[t] * dt + vol_path[t] * np.sqrt(dt) * z
            levels[t + 1] = np.maximum(levels[t + 1], 1e-12)
        # Map CIR level to a synthetic price via exponential for OHLCV compatibility
        prices = 100.0 * np.exp(levels - levels[0])
        rets = np.diff(np.log(np.clip(prices, 1e-12, None)), axis=0)
        return PathResult(
            prices=_squeeze(prices),
            returns=_squeeze(rets),
            volatility=_squeeze(vol_path),
            drift=_squeeze(drift_path),
            latent={"cir_level": _squeeze(levels)},
            metadata={"model": "cir"},
        )
