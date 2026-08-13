"""Particle resampling schemes."""

from __future__ import annotations

from typing import Literal

import numpy as np

from iqrp.app.math.probability.sampling import systematic_resample
from iqrp.app.regimes.particle.particle import ParticleCloud
from iqrp.app.regimes.particle.weighting import effective_sample_size

ResampleMethod = Literal["multinomial", "systematic", "residual", "stratified"]


def multinomial_resample(
    weights: np.ndarray, *, rng: np.random.Generator
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / max(float(w.sum()), 1e-300)
    n = w.size
    return rng.choice(n, size=n, replace=True, p=w).astype(np.int64)


def residual_resample(weights: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / max(float(w.sum()), 1e-300)
    n = w.size
    expected = n * w
    floor = np.floor(expected).astype(np.int64)
    indices = np.repeat(np.arange(n), floor)
    residual = expected - floor
    n_remain = n - int(floor.sum())
    if n_remain > 0:
        residual = residual / max(float(residual.sum()), 1e-300)
        extra = rng.choice(n, size=n_remain, replace=True, p=residual)
        indices = np.concatenate([indices, extra])
    # pad/truncate for safety
    if indices.size < n:
        pad = rng.choice(n, size=n - indices.size, replace=True, p=w)
        indices = np.concatenate([indices, pad])
    return indices[:n].astype(np.int64)


def stratified_resample(weights: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / max(float(w.sum()), 1e-300)
    n = w.size
    positions = (np.arange(n) + rng.random(n)) / n
    cumsum = np.cumsum(w)
    return np.searchsorted(cumsum, positions).astype(np.int64)


def resample_indices(
    weights: np.ndarray,
    method: ResampleMethod = "systematic",
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    gen = rng or np.random.default_rng()
    if method == "multinomial":
        return multinomial_resample(weights, rng=gen)
    if method == "residual":
        return residual_resample(weights, rng=gen)
    if method == "stratified":
        return stratified_resample(weights, rng=gen)
    return systematic_resample(weights, rng=gen)


def apply_resampling(
    cloud: ParticleCloud,
    method: ResampleMethod = "systematic",
    *,
    rng: np.random.Generator | None = None,
) -> ParticleCloud:
    idx = resample_indices(cloud.weights, method=method, rng=rng)
    n = cloud.n_particles
    log_w = np.full(n, -np.log(n), dtype=np.float64)
    return ParticleCloud(
        states=cloud.states[idx].copy(),
        log_weights=log_w,
        likelihoods=cloud.likelihoods[idx].copy(),
        timestamps=(
            None if cloud.timestamps is None else cloud.timestamps[idx].copy()
        ),
        metadata={**cloud.metadata, "resample_indices": idx.tolist()},
    )


def adaptive_resample(
    cloud: ParticleCloud,
    *,
    ess_threshold: float = 0.5,
    method: ResampleMethod = "systematic",
    rng: np.random.Generator | None = None,
) -> tuple[ParticleCloud, bool]:
    """Resample if ESS < threshold * N. Returns ``(cloud, did_resample)``."""
    ess = effective_sample_size(cloud.weights)
    thresh = float(ess_threshold) * cloud.n_particles
    if ess < thresh:
        return apply_resampling(cloud, method=method, rng=rng), True
    return cloud, False
