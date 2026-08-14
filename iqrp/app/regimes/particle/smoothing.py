"""Particle smoothing (forward-filter backward-simulator style)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.regimes.particle.particle import FilterTrace
from iqrp.app.regimes.particle.propagation import TransitionModel


@dataclass
class SmoothTrace:
    means: np.ndarray
    covs: np.ndarray
    trajectories: np.ndarray  # (N, T, d) sampled trajectories
    metadata: dict[str, Any] = field(default_factory=dict)


def _transition_logpdf(
    x_next: np.ndarray,
    x_prev: np.ndarray,
    model: TransitionModel,
) -> np.ndarray:
    """Approximate Gaussian transition density log p(x_t | x_{t-1})."""
    # mean under linear F if available else identity
    if model.f is not None:
        mean = x_prev @ np.asarray(model.f, dtype=np.float64).T
    else:
        mean = x_prev
    resid = x_next - mean
    var = max(model.q_scale**2, 1e-12)
    # isotropic Gaussian
    d = resid.shape[-1] if resid.ndim > 1 else 1
    quad = np.sum(resid**2, axis=-1) / var
    return -0.5 * (d * np.log(2 * np.pi * var) + quad)


def trajectory_smooth(
    trace: FilterTrace,
    model: TransitionModel,
    *,
    n_trajectories: int | None = None,
    rng: np.random.Generator | None = None,
) -> SmoothTrace:
    """
    Backward trajectory sampling (FFBSi-lite).

    Samples full trajectories from the filtered particle clouds.
    """
    gen = rng or np.random.default_rng()
    clouds = trace.clouds
    t_steps = len(clouds)
    if t_steps == 0:
        return SmoothTrace(
            means=np.zeros((0, 1)),
            covs=np.zeros((0, 1, 1)),
            trajectories=np.zeros((0, 0, 1)),
        )
    n = int(n_trajectories if n_trajectories is not None else clouds[-1].n_particles)
    d = clouds[0].dim
    traj = np.empty((n, t_steps, d), dtype=np.float64)

    # sample at final time
    w_T = clouds[-1].weights
    idx = gen.choice(clouds[-1].n_particles, size=n, replace=True, p=w_T)
    traj[:, -1, :] = clouds[-1].states[idx]

    for t in range(t_steps - 2, -1, -1):
        cloud_t = clouds[t]
        # for each trajectory, compute backward weights ∝ w_t * p(x_{t+1}|x_t)
        x_next = traj[:, t + 1, :]
        # vectorized over ancestors: for each traj, score all particles at t
        # memory: for large N use loop over trajectories in batches
        ancestors = np.empty(n, dtype=np.int64)
        for i in range(n):
            log_p = _transition_logpdf(x_next[i], cloud_t.states, model)
            log_bw = np.log(np.clip(cloud_t.weights, 1e-300, None)) + log_p
            log_bw = log_bw - np.max(log_bw)
            bw = np.exp(log_bw)
            bw = bw / max(float(bw.sum()), 1e-300)
            ancestors[i] = int(gen.choice(cloud_t.n_particles, p=bw))
        traj[:, t, :] = cloud_t.states[ancestors]

    means = traj.mean(axis=0)
    covs = np.empty((t_steps, d, d), dtype=np.float64)
    for t in range(t_steps):
        covs[t] = np.cov(traj[:, t, :].T) if d > 1 else np.array([[float(np.var(traj[:, t, 0]))]])
        if covs[t].ndim == 0:
            covs[t] = np.array([[float(covs[t])]])
    return SmoothTrace(means=means, covs=covs, trajectories=traj, metadata={"method": "ffbs"})


def backward_smooth_means(trace: FilterTrace) -> tuple[np.ndarray, np.ndarray]:
    """Cheap smoother: use filtered means/covs (identity for bootstrap storage)."""
    return trace.means.copy(), trace.covs.copy()
