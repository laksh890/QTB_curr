"""Ornstein-Uhlenbeck mean-reverting process."""

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
class OrnsteinUhlenbeck(PathGenerator):
    meta = GeneratorMeta(
        name="ou",
        version="1.0.0",
        description="Ornstein-Uhlenbeck mean-reverting process",
        family="mean_reversion",
        parameters={"kappa": 1.0, "theta": 100.0, "volatility": 5.0},
    )

    def generate(
        self,
        n_steps: int,
        *,
        x0: float | np.ndarray = 100.0,
        dt: float = 0.004,
        mean_reversion_speed: float = 1.0,
        mean_reversion_level: float = 100.0,
        volatility: float | np.ndarray = 5.0,
        noise: NoiseName = "gaussian",
        correlation: np.ndarray | None = None,
        noise_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> PathResult:
        x0_arr = np.atleast_1d(np.asarray(x0, dtype=np.float64))
        n_assets = x0_arr.size
        kappa = float(mean_reversion_speed)
        theta = float(mean_reversion_level)
        sig = as_path_matrix(volatility, n_steps, n_assets)
        z = _innovations(n_steps, n_assets, self.rng, noise, correlation, noise_kwargs or {})
        levels = np.zeros((n_steps + 1, n_assets), dtype=np.float64)
        levels[0] = x0_arr
        drift_path = np.zeros((n_steps, n_assets), dtype=np.float64)
        for t in range(n_steps):
            drift_path[t] = kappa * (theta - levels[t])
            levels[t + 1] = levels[t] + drift_path[t] * dt + sig[t] * np.sqrt(dt) * z[t]
        levels = np.maximum(levels, 1e-8)
        rets = np.diff(np.log(levels), axis=0)
        return PathResult(
            prices=_squeeze(levels),
            returns=_squeeze(rets),
            volatility=_squeeze(sig),
            drift=_squeeze(drift_path),
            metadata={"model": "ou", "kappa": kappa, "theta": theta},
        )
