"""Merton jump-diffusion process."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.simulation.base.generator import (
    GeneratorMeta,
    PathGenerator,
    PathResult,
    register_generator,
)
from iqrp.app.simulation.noise.distributions import NoiseName
from iqrp.app.simulation.stochastic.gbm import _innovations, _squeeze, as_path_matrix


@register_generator
class MertonJumpDiffusion(PathGenerator):
    """Merton (1976) jump-diffusion: GBM + compound Poisson log-normal jumps."""

    meta = GeneratorMeta(
        name="merton_jump",
        version="1.0.0",
        description="Merton jump-diffusion process",
        family="jump_diffusion",
        parameters={
            "drift": 0.05,
            "volatility": 0.2,
            "jump_intensity": 5.0,
            "jump_mean": -0.02,
            "jump_std": 0.04,
        },
    )

    def generate(
        self,
        n_steps: int,
        *,
        x0: float | np.ndarray = 100.0,
        dt: float = 0.004,
        drift: float | np.ndarray = 0.05,
        volatility: float | np.ndarray = 0.2,
        jump_intensity: float = 5.0,
        jump_mean: float = -0.02,
        jump_std: float = 0.04,
        noise: NoiseName = "gaussian",
        correlation: np.ndarray | None = None,
        noise_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> PathResult:
        x0_arr = np.atleast_1d(np.asarray(x0, dtype=np.float64))
        n_assets = x0_arr.size
        mu = as_path_matrix(drift, n_steps, n_assets)
        sig = as_path_matrix(volatility, n_steps, n_assets)
        z = _innovations(n_steps, n_assets, self.rng, noise, correlation, noise_kwargs or {})
        lam = float(jump_intensity)
        # Compensator for jump mean: λ (E[e^J]-1)
        jump_comp = lam * (np.exp(jump_mean + 0.5 * jump_std**2) - 1.0)
        n_jumps = self.rng.poisson(lam * dt, size=(n_steps, n_assets))
        jump_sizes = np.zeros((n_steps, n_assets), dtype=np.float64)
        for t in range(n_steps):
            for a in range(n_assets):
                k = int(n_jumps[t, a])
                if k > 0:
                    jump_sizes[t, a] = float(np.sum(self.rng.normal(jump_mean, jump_std, size=k)))
        rets = (mu - 0.5 * sig**2 - jump_comp) * dt + sig * np.sqrt(dt) * z + jump_sizes
        log_prices = np.log(np.clip(x0_arr, 1e-12, None)) + np.cumsum(rets, axis=0)
        prices = np.vstack([x0_arr, np.exp(log_prices)])
        return PathResult(
            prices=_squeeze(prices),
            returns=_squeeze(rets),
            volatility=_squeeze(sig),
            drift=_squeeze(mu),
            latent={
                "n_jumps": _squeeze(n_jumps.astype(np.float64)),
                "jump_sizes": _squeeze(jump_sizes),
            },
            metadata={"model": "merton_jump", "jump_intensity": lam},
        )


@register_generator
class JumpDiffusion(MertonJumpDiffusion):
    """Alias registered as ``jump_diffusion`` (same as Merton)."""

    meta = GeneratorMeta(
        name="jump_diffusion",
        version="1.0.0",
        description="Jump-diffusion alias for Merton jump process",
        family="jump_diffusion",
        parameters=dict(MertonJumpDiffusion.meta.parameters),
    )
