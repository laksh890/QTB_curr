"""Adaptive particle count and proposal control."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.regimes.particle.config import ParticleSettings
from iqrp.app.regimes.particle.particle import ParticleCloud
from iqrp.app.regimes.particle.resampling import apply_resampling
from iqrp.app.regimes.particle.weighting import effective_sample_size


def suggest_n_particles(
    ess: float,
    current_n: int,
    *,
    min_particles: int = 100,
    max_particles: int = 5000,
    target_ess_fraction: float = 0.5,
) -> int:
    """Heuristic: grow N when ESS fraction is low, shrink when high."""
    frac = ess / max(current_n, 1)
    target = float(target_ess_fraction)
    if frac < target * 0.5:
        n = int(current_n * 1.5)
    elif frac > target * 1.5:
        n = int(current_n * 0.75)
    else:
        n = current_n
    return int(np.clip(n, min_particles, max_particles))


def resize_cloud(
    cloud: ParticleCloud,
    n_new: int,
    *,
    rng: np.random.Generator,
) -> ParticleCloud:
    """Resample cloud to a new particle count."""
    n_new = max(1, int(n_new))
    w = cloud.weights
    idx = rng.choice(cloud.n_particles, size=n_new, replace=True, p=w)
    log_w = np.full(n_new, -np.log(n_new), dtype=np.float64)
    return ParticleCloud(
        states=cloud.states[idx].copy(),
        log_weights=log_w,
        likelihoods=cloud.likelihoods[idx].copy(),
        timestamps=(None if cloud.timestamps is None else cloud.timestamps[idx].copy()),
        metadata={**cloud.metadata, "resized_to": n_new},
    )


def adaptive_step(
    cloud: ParticleCloud,
    settings: ParticleSettings,
    *,
    rng: np.random.Generator,
) -> tuple[ParticleCloud, dict[str, Any]]:
    """Optionally resample and resize based on ESS."""
    info: dict[str, Any] = {"resized": False, "resampled": False}
    ess = effective_sample_size(cloud.weights)
    info["ess"] = ess
    thresh = settings.resampling.ess_threshold * cloud.n_particles
    out = cloud
    if settings.resampling.adaptive and ess < thresh:
        out = apply_resampling(out, method=settings.resampling.method, rng=rng)
        info["resampled"] = True
    if settings.adaptive.enabled:
        n_new = suggest_n_particles(
            ess,
            out.n_particles,
            min_particles=settings.adaptive.min_particles,
            max_particles=settings.adaptive.max_particles,
            target_ess_fraction=settings.adaptive.target_ess_fraction,
        )
        if n_new != out.n_particles:
            out = resize_cloud(out, n_new, rng=rng)
            info["resized"] = True
            info["n_particles"] = n_new
    return out, info
