"""Particle rejuvenation (MCMC / jitter) after resampling."""

from __future__ import annotations

from typing import Literal

import numpy as np

from iqrp.app.regimes.particle.particle import ParticleCloud
from iqrp.app.regimes.particle.propagation import TransitionModel
from iqrp.app.regimes.particle.weighting import log_likelihood

RejuvenationMethod = Literal["jitter", "mcmc", "adaptive", "covariance"]


def gaussian_jitter(
    cloud: ParticleCloud,
    *,
    scale: float = 0.05,
    rng: np.random.Generator,
) -> ParticleCloud:
    noise = rng.normal(0.0, scale, size=cloud.states.shape)
    return ParticleCloud(
        states=cloud.states + noise,
        log_weights=cloud.log_weights.copy(),
        likelihoods=cloud.likelihoods.copy(),
        timestamps=None if cloud.timestamps is None else cloud.timestamps.copy(),
        metadata={**cloud.metadata, "rejuvenation": "jitter"},
    )


def covariance_perturbation(
    cloud: ParticleCloud,
    *,
    scale: float = 0.05,
    rng: np.random.Generator,
) -> ParticleCloud:
    cov = cloud.covariance()
    # regularize
    cov = cov + 1e-9 * np.eye(cov.shape[0])
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        chol = np.diag(np.sqrt(np.clip(np.diag(cov), 1e-12, None)))
    z = rng.normal(0.0, 1.0, size=cloud.states.shape)
    noise = (z @ chol.T) * scale
    return ParticleCloud(
        states=cloud.states + noise,
        log_weights=cloud.log_weights.copy(),
        likelihoods=cloud.likelihoods.copy(),
        timestamps=None if cloud.timestamps is None else cloud.timestamps.copy(),
        metadata={**cloud.metadata, "rejuvenation": "covariance"},
    )


def adaptive_perturbation(
    cloud: ParticleCloud,
    *,
    scale: float = 0.05,
    rng: np.random.Generator,
) -> ParticleCloud:
    # scale by inverse ESS fraction (more jitter when degenerate)
    ess_frac = cloud.ess() / max(cloud.n_particles, 1)
    adapt_scale = scale * (1.0 + (1.0 - ess_frac))
    return gaussian_jitter(cloud, scale=adapt_scale, rng=rng)


def mcmc_rejuvenation(
    cloud: ParticleCloud,
    model: TransitionModel,
    observation: np.ndarray,
    *,
    scale: float = 0.05,
    steps: int = 1,
    obs_scale: float = 0.1,
    kind: str = "gaussian",
    df: float = 5.0,
    rng: np.random.Generator,
) -> ParticleCloud:
    """Random-walk Metropolis rejuvenation using observation likelihood."""
    states = cloud.states.copy()
    for _ in range(max(1, steps)):
        proposal = states + rng.normal(0.0, scale, size=states.shape)
        y_cur = model.observe(states)
        y_prop = model.observe(proposal)
        ll_cur = log_likelihood(observation, y_cur, scale=obs_scale, kind=kind, df=df)  # type: ignore[arg-type]
        ll_prop = log_likelihood(observation, y_prop, scale=obs_scale, kind=kind, df=df)  # type: ignore[arg-type]
        log_alpha = ll_prop - ll_cur
        accept = np.log(rng.random(states.shape[0])) < log_alpha
        states[accept] = proposal[accept]
    return ParticleCloud(
        states=states,
        log_weights=cloud.log_weights.copy(),
        likelihoods=cloud.likelihoods.copy(),
        timestamps=None if cloud.timestamps is None else cloud.timestamps.copy(),
        metadata={**cloud.metadata, "rejuvenation": "mcmc"},
    )


def rejuvenate(
    cloud: ParticleCloud,
    *,
    method: RejuvenationMethod = "jitter",
    scale: float = 0.05,
    model: TransitionModel | None = None,
    observation: np.ndarray | None = None,
    mcmc_steps: int = 1,
    obs_scale: float = 0.1,
    kind: str = "gaussian",
    df: float = 5.0,
    rng: np.random.Generator | None = None,
) -> ParticleCloud:
    gen = rng or np.random.default_rng()
    if method == "covariance":
        return covariance_perturbation(cloud, scale=scale, rng=gen)
    if method == "adaptive":
        return adaptive_perturbation(cloud, scale=scale, rng=gen)
    if method == "mcmc" and model is not None and observation is not None:
        return mcmc_rejuvenation(
            cloud,
            model,
            observation,
            scale=scale,
            steps=mcmc_steps,
            obs_scale=obs_scale,
            kind=kind,
            df=df,
            rng=gen,
        )
    return gaussian_jitter(cloud, scale=scale, rng=gen)
