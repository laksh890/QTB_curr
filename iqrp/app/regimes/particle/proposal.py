"""Importance proposal distributions for particle filters."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.regimes.particle.particle import ParticleCloud
from iqrp.app.regimes.particle.propagation import TransitionModel, propagate_cloud


def bootstrap_proposal(
    cloud: ParticleCloud,
    model: TransitionModel,
    *,
    rng: np.random.Generator,
    t: int = 0,
) -> tuple[ParticleCloud, np.ndarray]:
    """Prior proposal: sample from transition; proposal ratio = 0 (absorbed in bootstrap)."""
    proposed = propagate_cloud(cloud, model, rng=rng, t=t)
    return proposed, np.zeros(cloud.n_particles, dtype=np.float64)


def adaptive_proposal(
    cloud: ParticleCloud,
    model: TransitionModel,
    observation: np.ndarray,
    *,
    rng: np.random.Generator,
    shrink: float = 0.5,
    t: int = 0,
) -> tuple[ParticleCloud, np.ndarray]:
    """
    Adaptive Gaussian proposal centered between prior predict and observation.

    Uses shrunk process noise and returns log proposal correction ≈ 0
    (approximation suitable for bootstrap-style weighting with adapted variance).
    """
    prior = propagate_cloud(cloud, model, rng=rng, t=t)
    z = np.asarray(observation, dtype=np.float64).reshape(-1)
    # pull primary state toward observation
    pulled = prior.states.copy()
    pulled[:, 0] = (1.0 - shrink) * prior.states[:, 0] + shrink * z[0]
    # extra jitter with reduced scale
    pulled = pulled + rng.normal(0.0, model.q_scale * (1.0 - 0.5 * shrink), size=pulled.shape)
    # approximate log density ratio as zero (self-normalized IS with adapted proposal)
    return (
        ParticleCloud(
            states=pulled,
            log_weights=cloud.log_weights.copy(),
            likelihoods=cloud.likelihoods.copy(),
            timestamps=None if cloud.timestamps is None else cloud.timestamps.copy(),
            metadata={**cloud.metadata, "proposal": "adaptive"},
        ),
        np.zeros(cloud.n_particles, dtype=np.float64),
    )


def auxiliary_first_stage_weights(
    cloud: ParticleCloud,
    model: TransitionModel,
    observation: np.ndarray,
    *,
    scale: float,
    kind: str = "gaussian",
    df: float = 5.0,
) -> np.ndarray:
    """APF first-stage weights using expected observation from current particles."""
    from iqrp.app.regimes.particle.weighting import log_likelihood

    y_hat = model.observe(cloud.states)
    # look-ahead: use current predicted observation as proxy for next
    ll = log_likelihood(observation, y_hat, scale=scale, kind=kind, df=df)  # type: ignore[arg-type]
    return cloud.log_weights + ll


def propose(
    cloud: ParticleCloud,
    model: TransitionModel,
    observation: np.ndarray | None,
    *,
    kind: str = "bootstrap",
    rng: np.random.Generator,
    t: int = 0,
    shrink: float = 0.5,
) -> tuple[ParticleCloud, np.ndarray]:
    if kind == "adaptive" and observation is not None:
        return adaptive_proposal(cloud, model, observation, rng=rng, shrink=shrink, t=t)
    return bootstrap_proposal(cloud, model, rng=rng, t=t)


def proposal_diagnostics(cloud: ParticleCloud) -> dict[str, Any]:
    return {
        "n_particles": cloud.n_particles,
        "state_mean": cloud.mean().tolist(),
        "state_std": np.sqrt(np.clip(np.diag(cloud.covariance()), 0, None)).tolist(),
        "proposal": cloud.metadata.get("proposal", "bootstrap"),
    }
