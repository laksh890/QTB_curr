"""Forward-backward: posteriors, expected transitions, occupancy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.math.utils.numerical_stability import stable_softmax
from iqrp.app.regimes.hmm.backward import backward
from iqrp.app.regimes.hmm.forward import forward
from iqrp.app.state_space.base.probabilities import state_occupancy_probabilities


@dataclass(frozen=True, slots=True)
class ForwardBackwardResult:
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    xi: np.ndarray
    scales: np.ndarray
    log_likelihood: float


def forward_backward(
    log_emissions: Any,
    transition: Any,
    *,
    initial: Any | None = None,
    eps: float = 1.0e-300,
) -> ForwardBackwardResult:
    log_b = np.asarray(log_emissions, dtype=np.float64)
    alpha, scales, ll = forward(log_b, transition, initial=initial, eps=eps)
    beta = backward(log_b, transition, scales=scales, eps=eps)
    gamma = state_occupancy_probabilities(alpha, beta)
    xi = expected_transitions(alpha, beta, log_b, transition, eps=eps)
    return ForwardBackwardResult(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        xi=xi,
        scales=scales,
        log_likelihood=ll,
    )


def expected_transitions(
    alpha: np.ndarray,
    beta: np.ndarray,
    log_emissions: np.ndarray,
    transition: Any,
    *,
    eps: float = 1.0e-300,
) -> np.ndarray:
    """Return ``xi`` of shape ``(T-1, K, K)``."""
    a = np.asarray(alpha, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    log_e = np.asarray(log_emissions, dtype=np.float64)
    p = normalize_rows(np.asarray(transition, dtype=np.float64))
    t_steps, k = a.shape
    log_p = np.log(np.clip(p, eps, None))
    xi = np.empty((max(t_steps - 1, 0), k, k), dtype=np.float64)
    for t in range(t_steps - 1):
        log_xi = (
            np.log(np.clip(a[t], eps, None))[:, None]
            + log_p
            + log_e[t + 1][None, :]
            + np.log(np.clip(b[t + 1], eps, None))[None, :]
        )
        xi[t] = stable_softmax(log_xi.reshape(-1), axis=0).reshape(k, k)
    return xi


def posterior_state_probabilities(result: ForwardBackwardResult) -> np.ndarray:
    return result.gamma


def expected_state_occupancy(result: ForwardBackwardResult) -> np.ndarray:
    return result.gamma.sum(axis=0)
