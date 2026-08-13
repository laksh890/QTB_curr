"""Forecast uncertainty summaries."""

from __future__ import annotations

from typing import Any

import numpy as np


def forecast_uncertainty(
    step_distributions: Any,
    *,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Entropy and max-prob bands across the forecast horizon."""
    steps = np.asarray(step_distributions, dtype=np.float64)
    if steps.ndim == 1:
        steps = steps.reshape(1, -1)
    # row-normalize
    steps = steps / np.clip(steps.sum(axis=1, keepdims=True), 1e-300, None)
    max_prob = steps.max(axis=1)
    # Shannon entropy in nats
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(np.clip(steps, 1e-300, None))
        entropy = -np.sum(steps * log_p, axis=1)
    # Approximate credible mass of top states
    sorted_p = np.sort(steps, axis=1)[:, ::-1]
    cum = np.cumsum(sorted_p, axis=1)
    top_k = (cum < confidence_level).sum(axis=1) + 1
    return {
        "max_probability": max_prob.tolist(),
        "entropy": entropy.tolist(),
        "top_k_for_confidence": top_k.astype(int).tolist(),
        "confidence_level": float(confidence_level),
        "mean_entropy": float(np.mean(entropy)),
        "terminal_max_probability": float(max_prob[-1]),
    }


def probability_interval(
    distribution: Any,
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Return (lo, hi) probability mass bounds of the HPD state set."""
    p = np.asarray(distribution, dtype=np.float64).reshape(-1)
    p = p / max(float(p.sum()), 1e-300)
    order = np.argsort(p)[::-1]
    cum = 0.0
    included: list[float] = []
    for i in order:
        included.append(float(p[i]))
        cum += float(p[i])
        if cum >= confidence_level:
            break
    if not included:
        return (0.0, 0.0)
    return (min(included), max(included))
