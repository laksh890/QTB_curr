"""Discrete random walk generators."""

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
class RandomWalk(PathGenerator):
    meta = GeneratorMeta(
        name="random_walk",
        version="1.0.0",
        description="Discrete random walk on log-prices",
        family="discrete",
        parameters={"volatility": 0.01},
    )

    def generate(
        self,
        n_steps: int,
        *,
        x0: float | np.ndarray = 100.0,
        dt: float = 1.0,
        volatility: float | np.ndarray = 0.01,
        drift: float | np.ndarray = 0.0,
        noise: NoiseName = "gaussian",
        correlation: np.ndarray | None = None,
        noise_kwargs: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> PathResult:
        del dt, _kwargs
        x0_arr = np.atleast_1d(np.asarray(x0, dtype=np.float64))
        n_assets = x0_arr.size
        mu = as_path_matrix(drift, n_steps, n_assets)
        sig = as_path_matrix(volatility, n_steps, n_assets)
        z = _innovations(n_steps, n_assets, self.rng, noise, correlation, noise_kwargs or {})
        rets = mu + sig * z
        log_prices = np.log(np.clip(x0_arr, 1e-12, None)) + np.cumsum(rets, axis=0)
        prices = np.vstack([x0_arr, np.exp(log_prices)])
        return PathResult(
            prices=_squeeze(prices),
            returns=_squeeze(rets),
            volatility=_squeeze(sig),
            drift=_squeeze(mu),
            metadata={"model": "random_walk"},
        )
