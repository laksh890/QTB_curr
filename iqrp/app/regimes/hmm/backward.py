"""Backward algorithm (scaled / log-space)."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.state_space.base.probabilities import backward_probabilities


def backward(
    log_emissions: Any,
    transition: Any,
    *,
    scales: Any | None = None,
    eps: float = 1.0e-300,
) -> np.ndarray:
    """Return scaled backward messages ``beta (T, K)``."""
    return backward_probabilities(log_emissions, transition, scales=scales, eps=eps)
