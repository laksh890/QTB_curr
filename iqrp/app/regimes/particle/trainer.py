"""Particle filter runners and training orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.regimes.particle.adaptive import adaptive_step
from iqrp.app.regimes.particle.config import ParticleSettings
from iqrp.app.regimes.particle.particle import FilterTrace, ParticleCloud
from iqrp.app.regimes.particle.propagation import TransitionModel, build_transition
from iqrp.app.regimes.particle.proposal import (
    auxiliary_first_stage_weights,
    bootstrap_proposal,
    propose,
)
from iqrp.app.regimes.particle.rejuvenation import rejuvenate
from iqrp.app.regimes.particle.resampling import adaptive_resample, apply_resampling
from iqrp.app.regimes.particle.weighting import log_likelihood, update_weights


@dataclass
class TrainResult:
    model: TransitionModel
    trace: FilterTrace
    history: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def initialize_cloud(
    n_particles: int,
    n_states: int,
    settings: ParticleSettings,
    *,
    rng: np.random.Generator,
    x0: np.ndarray | None = None,
) -> ParticleCloud:
    scale = float(settings.system.initial_covariance_scale) ** 0.5
    center = (
        np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0 is not None
        else np.zeros(n_states)
    )
    if center.size < n_states:
        center = np.pad(center, (0, n_states - center.size))
    states = center[:n_states] + rng.normal(0.0, scale, size=(n_particles, n_states))
    if x0 is None and settings.system.initial_state_scale != 0:
        states = states + float(settings.system.initial_state_scale) * 0.0
    return ParticleCloud.equal_weight(states)


def simulate_nonlinear(
    model: TransitionModel,
    n_steps: int,
    *,
    rng: np.random.Generator | None = None,
    x0: np.ndarray | None = None,
    obs_scale: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate latent states and nonlinear observations."""
    gen = rng or np.random.default_rng()
    d = model.n_states
    x = np.zeros(d) if x0 is None else np.asarray(x0, dtype=np.float64).reshape(-1)[:d]
    states = np.empty((n_steps, d), dtype=np.float64)
    obs = np.empty((n_steps, 1), dtype=np.float64)
    for t in range(n_steps):
        x = model.propagate(x.reshape(1, -1), rng=gen, t=t)[0]
        y = model.observe(x.reshape(1, -1))[0]
        states[t] = x
        obs[t] = y + gen.normal(0.0, obs_scale, size=y.shape)
    return states, obs


def _weight_and_stats(
    cloud: ParticleCloud,
    observation: np.ndarray,
    model: TransitionModel,
    settings: ParticleSettings,
    log_proposal_ratio: np.ndarray | None = None,
) -> ParticleCloud:
    y_hat = model.observe(cloud.states)
    ll = log_likelihood(
        observation,
        y_hat,
        scale=settings.system.observation_noise_scale,
        kind=settings.likelihood,
        df=settings.system.student_t_df,
    )
    return update_weights(cloud, ll, log_proposal_ratio=log_proposal_ratio)


def filter_bootstrap(
    observations: np.ndarray,
    model: TransitionModel,
    settings: ParticleSettings,
    *,
    rng: np.random.Generator,
    cloud0: ParticleCloud | None = None,
) -> FilterTrace:
    """Bootstrap particle filter (SIR with prior proposal)."""
    return _run_sir(
        observations,
        model,
        settings,
        rng=rng,
        cloud0=cloud0,
        proposal_kind="bootstrap",
        always_resample=False,
    )


def filter_sis(
    observations: np.ndarray,
    model: TransitionModel,
    settings: ParticleSettings,
    *,
    rng: np.random.Generator,
    cloud0: ParticleCloud | None = None,
) -> FilterTrace:
    """Sequential importance sampling without resampling."""
    z = np.asarray(observations, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n = settings.n_particles
    d = model.n_states
    cloud = cloud0 or initialize_cloud(n, d, settings, rng=rng, x0=z[0])
    means, covs, clouds, ess_hist, resamp = [], [], [], [], []
    ll = 0.0
    for t in range(z.shape[0]):
        cloud, ratio = bootstrap_proposal(cloud, model, rng=rng, t=t)
        cloud = _weight_and_stats(cloud, z[t], model, settings, ratio)
        ll += cloud.log_likelihood_increment()
        means.append(cloud.mean())
        covs.append(cloud.covariance())
        clouds.append(cloud)
        ess_hist.append(cloud.ess())
        resamp.append(False)
    return FilterTrace(
        means=np.asarray(means),
        covs=np.asarray(covs),
        clouds=clouds,
        ess=np.asarray(ess_hist, dtype=np.float64),
        resampled=np.asarray(resamp, dtype=bool),
        log_likelihood=ll,
        metadata={"filter": "sis"},
    )


def filter_sir(
    observations: np.ndarray,
    model: TransitionModel,
    settings: ParticleSettings,
    *,
    rng: np.random.Generator,
    cloud0: ParticleCloud | None = None,
) -> FilterTrace:
    """Sequential importance resampling (always resample)."""
    return _run_sir(
        observations,
        model,
        settings,
        rng=rng,
        cloud0=cloud0,
        proposal_kind="bootstrap",
        always_resample=True,
    )


def _run_sir(
    observations: np.ndarray,
    model: TransitionModel,
    settings: ParticleSettings,
    *,
    rng: np.random.Generator,
    cloud0: ParticleCloud | None,
    proposal_kind: str,
    always_resample: bool,
) -> FilterTrace:
    z = np.asarray(observations, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n = settings.n_particles
    d = model.n_states
    cloud = cloud0 or initialize_cloud(n, d, settings, rng=rng, x0=z[0])
    means, covs, clouds, ess_hist, resamp = [], [], [], [], []
    ll = 0.0
    for t in range(z.shape[0]):
        cloud, ratio = propose(
            cloud, model, z[t], kind=proposal_kind, rng=rng, t=t
        )
        cloud = _weight_and_stats(cloud, z[t], model, settings, ratio)
        ll += cloud.log_likelihood_increment()
        did = False
        if always_resample:
            cloud = apply_resampling(cloud, method=settings.resampling.method, rng=rng)
            did = True
        elif settings.resampling.adaptive:
            cloud, did = adaptive_resample(
                cloud,
                ess_threshold=settings.resampling.ess_threshold,
                method=settings.resampling.method,
                rng=rng,
            )
        if did and settings.rejuvenation.enabled:
            cloud = rejuvenate(
                cloud,
                method=settings.rejuvenation.method,
                scale=settings.rejuvenation.scale,
                model=model,
                observation=z[t],
                mcmc_steps=settings.rejuvenation.mcmc_steps,
                obs_scale=settings.system.observation_noise_scale,
                kind=settings.likelihood,
                df=settings.system.student_t_df,
                rng=rng,
            )
        means.append(cloud.mean())
        covs.append(cloud.covariance())
        clouds.append(cloud)
        ess_hist.append(cloud.ess())
        resamp.append(did)
    return FilterTrace(
        means=np.asarray(means),
        covs=np.asarray(covs),
        clouds=clouds,
        ess=np.asarray(ess_hist, dtype=np.float64),
        resampled=np.asarray(resamp, dtype=bool),
        log_likelihood=ll,
        metadata={"filter": "sir" if always_resample else "bootstrap"},
    )


def filter_auxiliary(
    observations: np.ndarray,
    model: TransitionModel,
    settings: ParticleSettings,
    *,
    rng: np.random.Generator,
    cloud0: ParticleCloud | None = None,
) -> FilterTrace:
    """Auxiliary particle filter (Pitt–Shephard)."""
    z = np.asarray(observations, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n = settings.n_particles
    d = model.n_states
    cloud = cloud0 or initialize_cloud(n, d, settings, rng=rng, x0=z[0])
    means, covs, clouds, ess_hist, resamp = [], [], [], [], []
    ll = 0.0
    for t in range(z.shape[0]):
        # first-stage weights
        log_a = auxiliary_first_stage_weights(
            cloud,
            model,
            z[t],
            scale=settings.system.observation_noise_scale,
            kind=settings.likelihood,
            df=settings.system.student_t_df,
        )
        from iqrp.app.math.utils.numerical_stability import stable_softmax

        w1 = stable_softmax(log_a)
        idx = rng.choice(cloud.n_particles, size=cloud.n_particles, replace=True, p=w1)
        parent = ParticleCloud(
            states=cloud.states[idx],
            log_weights=np.full(cloud.n_particles, -np.log(cloud.n_particles)),
            likelihoods=cloud.likelihoods[idx],
        )
        # propagate
        child, _ = bootstrap_proposal(parent, model, rng=rng, t=t)
        y_hat = model.observe(child.states)
        ll_t = log_likelihood(
            z[t],
            y_hat,
            scale=settings.system.observation_noise_scale,
            kind=settings.likelihood,
            df=settings.system.student_t_df,
        )
        # second-stage correction: subtract first-stage look-ahead contribution
        y_parent = model.observe(parent.states)
        ll_look = log_likelihood(
            z[t],
            y_parent,
            scale=settings.system.observation_noise_scale,
            kind=settings.likelihood,
            df=settings.system.student_t_df,
        )
        cloud = update_weights(child, ll_t - ll_look)
        ll += cloud.log_likelihood_increment()
        cloud, did = adaptive_resample(
            cloud,
            ess_threshold=settings.resampling.ess_threshold,
            method=settings.resampling.method,
            rng=rng,
        )
        if did and settings.rejuvenation.enabled:
            cloud = rejuvenate(
                cloud,
                method=settings.rejuvenation.method,
                scale=settings.rejuvenation.scale,
                model=model,
                observation=z[t],
                rng=rng,
            )
        means.append(cloud.mean())
        covs.append(cloud.covariance())
        clouds.append(cloud)
        ess_hist.append(cloud.ess())
        resamp.append(did)
    return FilterTrace(
        means=np.asarray(means),
        covs=np.asarray(covs),
        clouds=clouds,
        ess=np.asarray(ess_hist, dtype=np.float64),
        resampled=np.asarray(resamp, dtype=bool),
        log_likelihood=ll,
        metadata={"filter": "auxiliary"},
    )


def filter_rao_blackwellized(
    observations: np.ndarray,
    model: TransitionModel,
    settings: ParticleSettings,
    *,
    rng: np.random.Generator,
    cloud0: ParticleCloud | None = None,
) -> FilterTrace:
    """
    Rao-Blackwellized PF for conditionally linear-Gaussian systems.

    Nonlinear component tracked by particles; linear component by per-particle KF.
    State layout: [nonlinear (d_nl), linear mean stacked in metadata].
    """
    z = np.asarray(observations, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n = settings.n_particles
    n_lin = max(1, int(settings.rao_blackwellized.n_linear))
    d_nl = max(1, model.n_states)
    q_k = float(settings.rao_blackwellized.kalman_process_noise)
    r_k = float(settings.rao_blackwellized.kalman_observation_noise)
    cloud = cloud0 or initialize_cloud(n, d_nl, settings, rng=rng, x0=z[0])
    # per-particle Kalman state
    lin_mean = np.zeros((n, n_lin), dtype=np.float64)
    lin_cov = np.stack([np.eye(n_lin) * settings.system.initial_covariance_scale for _ in range(n)])

    means, covs, clouds, ess_hist, resamp = [], [], [], [], []
    ll = 0.0
    for t in range(z.shape[0]):
        # propagate nonlinear
        cloud, _ = bootstrap_proposal(cloud, model, rng=rng, t=t)
        # KF predict/update per particle (scalar observation coupling to linear state)
        for i in range(n):
            # predict
            lin_mean[i] = lin_mean[i]  # identity transition
            lin_cov[i] = lin_cov[i] + q_k * np.eye(n_lin)
            # observation: z ≈ observe(nl) + H lin
            h = np.ones((1, n_lin))
            y_nl = float(np.asarray(model.observe(cloud.states[i : i + 1])).reshape(-1)[0])
            innov = float(z[t, 0] - y_nl - float(np.asarray(h @ lin_mean[i]).reshape(-1)[0]))
            s = float(np.asarray(h @ lin_cov[i] @ h.T + r_k).reshape(-1)[0])
            k = (lin_cov[i] @ h.T) / max(s, 1e-12)
            lin_mean[i] = lin_mean[i] + (k.reshape(-1) * innov)
            lin_cov[i] = (np.eye(n_lin) - k @ h) @ lin_cov[i]
            # particle weight from innovation likelihood
        y_hat = model.observe(cloud.states)[:, 0] + lin_mean[:, 0]
        ll_t = log_likelihood(
            z[t],
            y_hat.reshape(-1, 1),
            scale=settings.system.observation_noise_scale,
            kind=settings.likelihood,
            df=settings.system.student_t_df,
        )
        cloud = update_weights(cloud, ll_t)
        ll += cloud.log_likelihood_increment()
        cloud, did = adaptive_resample(
            cloud,
            ess_threshold=settings.resampling.ess_threshold,
            method=settings.resampling.method,
            rng=rng,
        )
        if did:
            idx = np.asarray(cloud.metadata.get("resample_indices", np.arange(n)), dtype=np.int64)
            if idx.size == n:
                lin_mean = lin_mean[idx]
                lin_cov = lin_cov[idx]
        # combined mean: nonlinear + linear primary
        comb = cloud.states.copy()
        comb[:, 0] = comb[:, 0] + lin_mean[:, 0]
        w = cloud.weights
        mean = np.sum(w[:, None] * comb, axis=0)
        diff = comb - mean
        cov = (w[:, None, None] * (diff[:, :, None] @ diff[:, None, :])).sum(axis=0)
        means.append(mean)
        covs.append(cov)
        cloud.metadata["lin_mean"] = lin_mean.tolist()
        clouds.append(cloud)
        ess_hist.append(cloud.ess())
        resamp.append(did)
    return FilterTrace(
        means=np.asarray(means),
        covs=np.asarray(covs),
        clouds=clouds,
        ess=np.asarray(ess_hist, dtype=np.float64),
        resampled=np.asarray(resamp, dtype=bool),
        log_likelihood=ll,
        metadata={"filter": "rao_blackwellized"},
    )


def filter_adaptive(
    observations: np.ndarray,
    model: TransitionModel,
    settings: ParticleSettings,
    *,
    rng: np.random.Generator,
    cloud0: ParticleCloud | None = None,
) -> FilterTrace:
    """Adaptive PF: ESS-triggered resampling, optional N adaptation, adaptive proposal."""
    z = np.asarray(observations, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    n = settings.n_particles
    d = model.n_states
    cloud = cloud0 or initialize_cloud(n, d, settings, rng=rng, x0=z[0])
    means, covs, clouds, ess_hist, resamp = [], [], [], [], []
    ll = 0.0
    proposal_kind = "adaptive" if settings.adaptive.proposal_adapt else "bootstrap"
    for t in range(z.shape[0]):
        cloud, ratio = propose(cloud, model, z[t], kind=proposal_kind, rng=rng, t=t)
        cloud = _weight_and_stats(cloud, z[t], model, settings, ratio)
        ll += cloud.log_likelihood_increment()
        cloud, info = adaptive_step(cloud, settings, rng=rng)
        if info.get("resampled") and settings.rejuvenation.enabled:
            cloud = rejuvenate(
                cloud,
                method=settings.rejuvenation.method,
                scale=settings.rejuvenation.scale,
                model=model,
                observation=z[t],
                rng=rng,
            )
        means.append(cloud.mean())
        covs.append(cloud.covariance())
        clouds.append(cloud)
        ess_hist.append(cloud.ess())
        resamp.append(bool(info.get("resampled")))
    return FilterTrace(
        means=np.asarray(means),
        covs=np.asarray(covs),
        clouds=clouds,
        ess=np.asarray(ess_hist, dtype=np.float64),
        resampled=np.asarray(resamp, dtype=bool),
        log_likelihood=ll,
        metadata={"filter": "adaptive", "final_n": clouds[-1].n_particles if clouds else n},
    )


def run_filter(
    observations: np.ndarray,
    model: TransitionModel,
    settings: ParticleSettings,
    *,
    rng: np.random.Generator,
    cloud0: ParticleCloud | None = None,
) -> FilterTrace:
    ft = settings.filter_type
    if ft == "sis":
        return filter_sis(observations, model, settings, rng=rng, cloud0=cloud0)
    if ft == "sir":
        return filter_sir(observations, model, settings, rng=rng, cloud0=cloud0)
    if ft == "auxiliary":
        return filter_auxiliary(observations, model, settings, rng=rng, cloud0=cloud0)
    if ft == "rao_blackwellized":
        return filter_rao_blackwellized(observations, model, settings, rng=rng, cloud0=cloud0)
    if ft == "adaptive":
        return filter_adaptive(observations, model, settings, rng=rng, cloud0=cloud0)
    return filter_bootstrap(observations, model, settings, rng=rng, cloud0=cloud0)


class ParticleTrainer:
    def __init__(self, settings: ParticleSettings | None = None) -> None:
        self.settings = settings or ParticleSettings.default()

    def build_model(
        self,
        *,
        application: str | None = None,
        n_states: int | None = None,
    ) -> TransitionModel:
        from typing import cast

        from iqrp.app.regimes.particle.propagation import Application

        return build_transition(
            self.settings,
            application=cast(Application | None, application),
            n_states=n_states,
        )

    def fit(
        self,
        observations: np.ndarray,
        *,
        model: TransitionModel | None = None,
        rng: np.random.Generator | None = None,
    ) -> TrainResult:
        gen = rng or np.random.default_rng(self.settings.random_seed)
        y = np.asarray(observations, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        mod = model or self.build_model(n_states=self.settings.n_states)
        history: list[float] = []
        trace = run_filter(y, mod, self.settings, rng=gen)
        history.append(trace.log_likelihood)
        for _ in range(1, max(1, self.settings.training.n_iterations)):
            trace = run_filter(y, mod, self.settings, rng=gen)
            history.append(trace.log_likelihood)
            if abs(history[-1] - history[-2]) < self.settings.training.tol:
                break
        return TrainResult(model=mod, trace=trace, history=history, metadata={"filter_type": self.settings.filter_type})
