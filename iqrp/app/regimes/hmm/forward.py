"""Forward algorithm (scaled / log-space)."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.state_space.base.probabilities import forward_probabilities


def forward(
    log_emissions: Any,
    transition: Any,
    *,
    initial: Any | None = None,
    eps: float = 1.0e-300,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(alpha, scales, log_likelihood)`` via math/state-space engine."""
    return forward_probabilities(log_emissions, transition, initial=initial, eps=eps)


def forward_log_likelihood(
    log_emissions: Any,
    transition: Any,
    *,
    initial: Any | None = None,
) -> float:
    _, _, ll = forward(log_emissions, transition, initial=initial)
    return float(ll)
