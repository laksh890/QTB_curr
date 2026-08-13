"""Log-space Viterbi decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows


@dataclass(frozen=True, slots=True)
class ViterbiResult:
    states: np.ndarray
    log_prob: float
    confidence: np.ndarray
    backpointers: np.ndarray


def viterbi(
    log_emissions: Any,
    transition: Any,
    *,
    initial: Any | None = None,
    eps: float = 1.0e-300,
) -> ViterbiResult:
    """Most likely state path with backpointers and per-step confidence."""
    log_b = np.asarray(log_emissions, dtype=np.float64)
    t_steps, k = log_b.shape
    p = normalize_rows(np.asarray(transition, dtype=np.float64))
    log_p = np.log(np.clip(p, eps, None))
    if initial is None:
        log_pi = np.full(k, -np.log(k))
    else:
        pi = np.asarray(initial, dtype=np.float64).reshape(-1)
        pi = pi / max(float(pi.sum()), eps)
        log_pi = np.log(np.clip(pi, eps, None))

    delta = np.empty((t_steps, k), dtype=np.float64)
    psi = np.zeros((t_steps, k), dtype=np.int64)
    delta[0] = log_pi + log_b[0]
    for t in range(1, t_steps):
        scores = delta[t - 1][:, None] + log_p
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(k)] + log_b[t]

    states = np.empty(t_steps, dtype=np.int64)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(t_steps - 2, -1, -1):
        states[t] = int(psi[t + 1, states[t + 1]])

    # Softmax confidence from delta at each step
    conf = np.empty(t_steps, dtype=np.float64)
    for t in range(t_steps):
        m = float(np.max(delta[t]))
        e = np.exp(delta[t] - m)
        conf[t] = float(e[states[t]] / max(float(e.sum()), eps))

    return ViterbiResult(
        states=states,
        log_prob=float(np.max(delta[-1])),
        confidence=conf,
        backpointers=psi,
    )
