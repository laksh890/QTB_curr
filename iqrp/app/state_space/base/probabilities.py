"""Probability utilities for discrete latent-state inference (math-engine backed)."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.math.stochastic.markov_utils import n_step_transition
from iqrp.app.math.utils.numerical_stability import logsumexp, stable_softmax


def forward_probabilities(
    log_emissions: Any,
    transition: Any,
    *,
    initial: Any | None = None,
    eps: float = 1.0e-300,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Scaled forward recursion.

    Returns
    -------
    alpha : (T, K) filtered / filtered-predictive occupancy after scaling
    scales : (T,) normalization constants ``c_t``
    log_likelihood : ``sum_t log c_t``
    """
    log_b = np.asarray(log_emissions, dtype=np.float64)
    if log_b.ndim != 2:
        raise ValueError("log_emissions must be shape (T, K)")
    t_steps, k = log_b.shape
    p = normalize_rows(np.asarray(transition, dtype=np.float64))
    if initial is None:
        log_pi = np.full(k, -np.log(k), dtype=np.float64)
    else:
        pi = np.asarray(initial, dtype=np.float64).reshape(-1)
        pi = pi / max(float(pi.sum()), eps)
        log_pi = np.log(np.clip(pi, eps, None))

    log_alpha = np.empty((t_steps, k), dtype=np.float64)
    scales = np.empty(t_steps, dtype=np.float64)

    log_alpha[0] = log_pi + log_b[0]
    log_c0 = float(logsumexp(log_alpha[0]))
    scales[0] = float(np.exp(log_c0))
    log_alpha[0] -= log_c0

    log_p = np.log(np.clip(p, eps, None))
    for t in range(1, t_steps):
        # log alpha_t(j) = log b_t(j) + logsumexp_i [log alpha_{t-1}(i) + log P_ij]
        prev = log_alpha[t - 1][:, None] + log_p
        log_alpha[t] = log_b[t] + np.asarray(logsumexp(prev, axis=0), dtype=np.float64)
        log_ct = float(logsumexp(log_alpha[t]))
        scales[t] = float(np.exp(log_ct))
        log_alpha[t] -= log_ct

    alpha = np.exp(log_alpha)
    alpha = normalize_rows(alpha)
    log_lik = float(np.sum(np.log(np.clip(scales, eps, None))))
    return alpha, scales, log_lik


def backward_probabilities(
    log_emissions: Any,
    transition: Any,
    *,
    scales: Any | None = None,
    eps: float = 1.0e-300,
) -> np.ndarray:
    """Scaled backward messages ``β`` with optional filter scales.

    Returns ``(T, K)`` backward probabilities (normalized per row for convenience).
    """
    log_b = np.asarray(log_emissions, dtype=np.float64)
    t_steps, k = log_b.shape
    p = normalize_rows(np.asarray(transition, dtype=np.float64))
    log_p = np.log(np.clip(p, eps, None))
    log_beta = np.zeros((t_steps, k), dtype=np.float64)

    scale_arr = None if scales is None else np.asarray(scales, dtype=np.float64).reshape(-1)

    for t in range(t_steps - 2, -1, -1):
        # log β_t(i) = logsumexp_j [log P_ij + log b_{t+1}(j) + log β_{t+1}(j)]
        nxt = log_p + (log_b[t + 1] + log_beta[t + 1])[None, :]
        log_beta[t] = np.asarray(logsumexp(nxt, axis=1), dtype=np.float64)
        if scale_arr is not None:
            log_beta[t] -= np.log(max(float(scale_arr[t + 1]), eps))

    beta = np.exp(log_beta - np.max(log_beta, axis=1, keepdims=True))
    return normalize_rows(beta)


def state_occupancy_probabilities(alpha: Any, beta: Any) -> np.ndarray:
    """Smoothed occupancy ``gamma_t(i) proportional to alpha_t(i) * beta_t(i)``."""
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    raw = a * b
    return normalize_rows(raw)


def transition_probabilities(
    alpha: Any,
    beta: Any,
    log_emissions: Any,
    transition: Any,
    *,
    eps: float = 1.0e-300,
) -> np.ndarray:
    """Expected pairwise transitions ``ξ`` averaged over time, shape ``(K, K)``."""
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    log_e = np.asarray(log_emissions, dtype=np.float64)
    p = normalize_rows(np.asarray(transition, dtype=np.float64))
    t_steps, k = a.shape
    xi_sum = np.zeros((k, k), dtype=np.float64)
    log_p = np.log(np.clip(p, eps, None))
    for t in range(t_steps - 1):
        # ξ_t(i,j) ∝ alpha_t(i) P_ij b_{t+1}(j) beta_{t+1}(j)
        log_xi = (
            np.log(np.clip(a[t], eps, None))[:, None]
            + log_p
            + log_e[t + 1][None, :]
            + np.log(np.clip(b[t + 1], eps, None))[None, :]
        )
        xi = stable_softmax(log_xi.reshape(-1), axis=0).reshape(k, k)
        xi_sum += xi
    return normalize_rows(xi_sum)


def joint_probabilities(alpha: Any, beta: Any) -> np.ndarray:
    """Alias for occupancy (joint filter/smoother marginals)."""
    return state_occupancy_probabilities(alpha, beta)


def forecast_distribution(current: Any, transition: Any, steps: int) -> np.ndarray:
    """``π P^{steps}`` using math-engine matrix exponentiation."""
    pi = np.asarray(current, dtype=np.float64).reshape(-1)
    pi = pi / max(float(pi.sum()), 1e-300)
    p_n = n_step_transition(transition, steps)
    return np.asarray(pi @ p_n, dtype=np.float64)
