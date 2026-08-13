"""Shared likelihood / FFBS helpers for Bayesian HMM inference."""

from __future__ import annotations

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.math.utils.numerical_stability import logsumexp, stable_softmax
from iqrp.app.regimes.bayesian.emissions import BayesianEmissions
from iqrp.app.regimes.bayesian.transitions import BayesianTransitions

try:
    from numba import njit as _njit  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False
    _njit = None
else:
    _HAS_NUMBA = True


def _ffbs_backward_sample(
    alpha: np.ndarray,
    log_trans: np.ndarray,
    rng_uniforms: np.ndarray,
) -> np.ndarray:
    t_steps, k = alpha.shape
    states = np.empty(t_steps, dtype=np.int64)
    # sample last state
    probs = alpha[t_steps - 1].copy()
    s = probs.sum()
    if s <= 0:
        probs[:] = 1.0 / k
    else:
        probs /= s
    u = rng_uniforms[t_steps - 1]
    cdf = 0.0
    states[t_steps - 1] = k - 1
    for j in range(k):
        cdf += probs[j]
        if u <= cdf:
            states[t_steps - 1] = j
            break
    for t in range(t_steps - 2, -1, -1):
        nxt = states[t + 1]
        scores = alpha[t] * np.exp(log_trans[:, nxt])
        s2 = scores.sum()
        if s2 <= 0:
            scores[:] = 1.0 / k
        else:
            scores /= s2
        u = rng_uniforms[t]
        cdf = 0.0
        states[t] = k - 1
        for j in range(k):
            cdf += scores[j]
            if u <= cdf:
                states[t] = j
                break
    return states


_ffbs_fast = _njit(cache=True)(_ffbs_backward_sample) if _HAS_NUMBA else _ffbs_backward_sample


def forward_filter(
    log_emissions: np.ndarray,
    transition: np.ndarray,
    initial: np.ndarray,
    *,
    eps: float = 1e-300,
) -> tuple[np.ndarray, float]:
    """Scaled forward probabilities (alpha) and log-likelihood."""
    log_b = np.asarray(log_emissions, dtype=np.float64)
    t_steps, k = log_b.shape
    p = normalize_rows(np.asarray(transition, dtype=np.float64))
    pi = np.asarray(initial, dtype=np.float64).reshape(-1)
    pi = pi / max(float(pi.sum()), eps)
    alpha = np.empty((t_steps, k), dtype=np.float64)
    scales = np.empty(t_steps, dtype=np.float64)
    alpha[0] = pi * np.exp(log_b[0] - np.max(log_b[0]))
    scales[0] = max(float(alpha[0].sum()), eps)
    alpha[0] /= scales[0]
    for t in range(1, t_steps):
        alpha[t] = (alpha[t - 1] @ p) * np.exp(log_b[t] - np.max(log_b[t]))
        scales[t] = max(float(alpha[t].sum()), eps)
        alpha[t] /= scales[t]
    float(np.sum(np.log(scales)) + np.sum(np.max(log_b, axis=1)))
    # correct: scales already include emission max? Use log-space for accuracy
    return alpha, _log_likelihood_logspace(log_b, p, pi, eps=eps)


def _log_likelihood_logspace(
    log_b: np.ndarray,
    p: np.ndarray,
    pi: np.ndarray,
    *,
    eps: float,
) -> float:
    t_steps, _k = log_b.shape
    log_p = np.log(np.clip(p, eps, None))
    log_alpha = np.log(np.clip(pi, eps, None)) + log_b[0]
    ll = float(logsumexp(log_alpha))
    log_alpha = log_alpha - ll
    for t in range(1, t_steps):
        log_alpha = logsumexp(log_alpha[:, None] + log_p, axis=0) + log_b[t]
        c = float(logsumexp(log_alpha))
        ll += c
        log_alpha = log_alpha - c
    return ll


def ffbs(
    log_emissions: np.ndarray,
    transition: np.ndarray,
    initial: np.ndarray,
    *,
    rng: np.random.Generator,
    eps: float = 1e-300,
) -> tuple[np.ndarray, float]:
    """Forward-filtering backward-sampling of latent states."""
    alpha, ll = forward_filter(log_emissions, transition, initial, eps=eps)
    p = normalize_rows(np.asarray(transition, dtype=np.float64))
    log_p = np.log(np.clip(p, eps, None))
    uniforms = np.asarray(rng.random(alpha.shape[0]), dtype=np.float64)
    states = _ffbs_fast(alpha, log_p, uniforms)
    return np.asarray(states, dtype=np.int64), ll


def _ffbs_python(alpha: np.ndarray, log_trans: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    t_steps, k = alpha.shape
    states = np.empty(t_steps, dtype=np.int64)
    probs = alpha[-1] / max(float(alpha[-1].sum()), 1e-300)
    states[-1] = int(np.searchsorted(np.cumsum(probs), uniforms[-1]))
    states[-1] = min(states[-1], k - 1)
    for t in range(t_steps - 2, -1, -1):
        scores = alpha[t] * np.exp(log_trans[:, states[t + 1]])
        scores = scores / max(float(scores.sum()), 1e-300)
        states[t] = int(np.searchsorted(np.cumsum(scores), uniforms[t]))
        states[t] = min(states[t], k - 1)
    return states


def log_joint(
    observations: np.ndarray,
    transitions: BayesianTransitions,
    emissions: BayesianEmissions,
    states: np.ndarray,
    *,
    eps: float = 1e-300,
) -> float:
    """Complete-data log joint (priors omitted when they cancel in MH ratios)."""
    y = np.asarray(observations, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    s = np.asarray(states, dtype=np.int64).reshape(-1)
    log_e = emissions.log_prob(y)
    ll = float(np.sum(log_e[np.arange(s.size), s]))
    p = np.clip(transitions.transition, eps, None)
    pi = np.clip(transitions.initial, eps, None)
    ll += float(np.log(pi[s[0]]))
    for t in range(s.size - 1):
        ll += float(np.log(p[s[t], s[t + 1]]))
    return ll


def smoothed_state_probabilities(
    log_emissions: np.ndarray,
    transition: np.ndarray,
    initial: np.ndarray,
    *,
    eps: float = 1e-300,
) -> tuple[np.ndarray, float]:
    """Forward-backward state posteriors (gamma)."""
    log_b = np.asarray(log_emissions, dtype=np.float64)
    t_steps, k = log_b.shape
    p = normalize_rows(np.asarray(transition, dtype=np.float64))
    pi = np.asarray(initial, dtype=np.float64).reshape(-1)
    pi = pi / max(float(pi.sum()), eps)
    log_p = np.log(np.clip(p, eps, None))
    log_alpha = np.empty((t_steps, k), dtype=np.float64)
    log_alpha[0] = np.log(np.clip(pi, eps, None)) + log_b[0]
    for t in range(1, t_steps):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_p, axis=0) + log_b[t]
    log_beta = np.zeros((t_steps, k), dtype=np.float64)
    for t in range(t_steps - 2, -1, -1):
        log_beta[t] = logsumexp(log_p + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
    log_gamma = log_alpha + log_beta
    gamma = stable_softmax(log_gamma, axis=1)
    ll = float(logsumexp(log_alpha[-1]))
    return gamma, ll
