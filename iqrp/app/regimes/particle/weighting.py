"""Likelihood evaluation and importance weight updates."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from iqrp.app.math.utils.numerical_stability import logsumexp, stable_softmax
from iqrp.app.regimes.particle.particle import ParticleCloud

LikelihoodName = Literal["gaussian", "student_t", "laplace", "custom"]


def log_likelihood(
    observations: np.ndarray,
    predicted: np.ndarray,
    *,
    scale: float = 0.1,
    kind: LikelihoodName = "gaussian",
    df: float = 5.0,
    custom_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Per-particle log-likelihood of observation given predicted observation."""
    z = np.asarray(observations, dtype=np.float64).reshape(-1)
    y = np.asarray(predicted, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    # use primary channel
    resid = z[0] - y[:, 0]
    scale = max(float(scale), 1e-12)
    if kind == "custom" and custom_fn is not None:
        return np.asarray(custom_fn(z, y), dtype=np.float64).reshape(-1)
    if kind == "student_t":
        return np.asarray(stats.t.logpdf(resid, df=df, loc=0.0, scale=scale), dtype=np.float64)
    if kind == "laplace":
        return np.asarray(stats.laplace.logpdf(resid, loc=0.0, scale=scale), dtype=np.float64)
    return np.asarray(stats.norm.logpdf(resid, loc=0.0, scale=scale), dtype=np.float64)


def update_weights(
    cloud: ParticleCloud,
    log_likes: np.ndarray,
    *,
    log_proposal_ratio: np.ndarray | None = None,
) -> ParticleCloud:
    """Multiply weights by likelihood (and optional proposal correction) in log-space."""
    ll = np.asarray(log_likes, dtype=np.float64).reshape(-1)
    log_w = cloud.log_weights + ll
    if log_proposal_ratio is not None:
        log_w = log_w + np.asarray(log_proposal_ratio, dtype=np.float64).reshape(-1)
    # stabilize
    log_w = log_w - logsumexp(log_w)
    likes = np.exp(np.clip(ll, -700, 700))
    return ParticleCloud(
        states=cloud.states.copy(),
        log_weights=log_w,
        likelihoods=likes,
        timestamps=None if cloud.timestamps is None else cloud.timestamps.copy(),
        metadata=dict(cloud.metadata),
    )


def normalize_weights(log_weights: np.ndarray) -> np.ndarray:
    return stable_softmax(log_weights)


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    s = float(w.sum())
    if s <= 0:
        return 0.0
    w = w / s
    return float(1.0 / np.sum(w**2))


def weight_entropy(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    s = float(w.sum())
    if s <= 0:
        return 0.0
    w = np.clip(w / s, 1e-300, None)
    return float(-np.sum(w * np.log(w)))


def weight_diagnostics(cloud: ParticleCloud) -> dict[str, float]:
    w = cloud.weights
    return {
        "ess": effective_sample_size(w),
        "entropy": weight_entropy(w),
        "max_weight": float(np.max(w)),
        "min_weight": float(np.min(w)),
        "n_particles": float(cloud.n_particles),
        "degeneracy_ratio": float(1.0 - effective_sample_size(w) / max(cloud.n_particles, 1)),
    }
