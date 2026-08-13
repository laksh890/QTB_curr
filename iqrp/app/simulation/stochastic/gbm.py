"""Geometric and Arithmetic Brownian Motion generators."""

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


def _innovations(
    n_steps: int,
    n_assets: int,
    rng: np.random.Generator,
    noise: NoiseName,
    correlation: np.ndarray | None,
    noise_kwargs: dict[str, Any],
) -> np.ndarray:
    z = sample_noise((n_steps, n_assets), noise, rng=rng, **noise_kwargs)
    if correlation is not None and n_assets > 1:
        chol = np.linalg.cholesky(correlation)
        z = z @ chol.T
    return z


def as_path_matrix(param: float | np.ndarray, n_steps: int, n_assets: int) -> np.ndarray:
    """Broadcast scalar / (T,) / (A,) / (T,A) parameters to shape (T, A)."""
    arr = np.asarray(param, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((n_steps, n_assets), float(arr))
    if arr.ndim == 1:
        if arr.size == n_steps:
            return np.tile(arr.reshape(n_steps, 1), (1, n_assets))
        if arr.size == n_assets:
            return np.tile(arr.reshape(1, n_assets), (n_steps, 1))
        if arr.size == 1:
            return np.full((n_steps, n_assets), float(arr[0]))
        raise ValueError(f"Cannot map parameter shape {arr.shape} to ({n_steps}, {n_assets})")
    if arr.shape == (n_steps, n_assets):
        return arr
    return np.broadcast_to(arr, (n_steps, n_assets)).astype(np.float64)


@register_generator
class GeometricBrownianMotion(PathGenerator):
    meta = GeneratorMeta(
        name="gbm",
        version="1.0.0",
        description="Geometric Brownian Motion (Black-Scholes)",
        family="diffusion",
        parameters={"drift": 0.05, "volatility": 0.2},
    )

    def generate(
        self,
        n_steps: int,
        *,
        x0: float | np.ndarray = 100.0,
        dt: float = 0.004,
        drift: float | np.ndarray = 0.05,
        volatility: float | np.ndarray = 0.2,
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
        rets = (mu - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * z
        log_prices = np.log(np.clip(x0_arr, 1e-12, None)) + np.cumsum(rets, axis=0)
        prices = np.vstack([x0_arr, np.exp(log_prices)])
        return PathResult(
            prices=_squeeze(prices),
            returns=_squeeze(rets),
            volatility=_squeeze(sig),
            drift=_squeeze(mu),
            metadata={"model": "gbm"},
        )


@register_generator
class ArithmeticBrownianMotion(PathGenerator):
    meta = GeneratorMeta(
        name="abm",
        version="1.0.0",
        description="Arithmetic Brownian Motion",
        family="diffusion",
        parameters={"drift": 0.0, "volatility": 1.0},
    )

    def generate(
        self,
        n_steps: int,
        *,
        x0: float | np.ndarray = 100.0,
        dt: float = 0.004,
        drift: float | np.ndarray = 0.0,
        volatility: float | np.ndarray = 1.0,
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
        increments = mu * dt + sig * np.sqrt(dt) * z
        levels = np.vstack([x0_arr, x0_arr + np.cumsum(increments, axis=0)])
        # Reflect near zero for price-like positivity when used as prices
        levels = np.maximum(levels, 1e-8)
        rets = np.diff(np.log(levels), axis=0)
        return PathResult(
            prices=_squeeze(levels),
            returns=_squeeze(rets),
            volatility=_squeeze(sig),
            drift=_squeeze(mu),
            metadata={"model": "abm"},
        )


def _squeeze(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr
