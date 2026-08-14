"""Forecast-derived expected returns with confidence shrinkage toward a prior."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__VERSION__ = "1.0.0"


def forecast_expected_returns(
    forecasts: Sequence[float] | np.ndarray,
    *,
    confidence: Sequence[float] | np.ndarray | None = None,
    prior: Sequence[float] | np.ndarray | None = None,
    uncertainty: Sequence[float] | np.ndarray | None = None,
    names: Sequence[str] | None = None,
    version: str = __VERSION__,
) -> dict[str, Any]:
    """Map forecast arrays to expected returns with confidence scaling.

    Confidence cannot invent certainty: low confidence shrinks forecasts toward
    the prior (default zeros). Optional ``uncertainty`` further dampens the
    active weight via ``1 / (1 + uncertainty)``.
    """
    f = np.asarray(forecasts, dtype=np.float64).reshape(-1)
    n = int(f.size)
    if prior is None:
        p = np.zeros(n, dtype=np.float64)
        prior_method = "zero"
    else:
        p = np.asarray(prior, dtype=np.float64).reshape(-1)
        if p.size != n:
            raise ValueError(f"prior length {p.size} != forecasts length {n}")
        prior_method = "provided"

    if confidence is None:
        c = np.ones(n, dtype=np.float64)
        confidence_method = "assumed_one"
    else:
        c = np.asarray(confidence, dtype=np.float64).reshape(-1)
        if c.size != n:
            raise ValueError(f"confidence length {c.size} != forecasts length {n}")
        c = np.clip(c, 0.0, 1.0)
        confidence_method = "provided"

    if uncertainty is not None:
        u = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
        if u.size != n:
            raise ValueError(f"uncertainty length {u.size} != forecasts length {n}")
        u = np.maximum(u, 0.0)
        # Extra dampening: high uncertainty reduces effective confidence
        c = c / (1.0 + u)
        c = np.clip(c, 0.0, 1.0)
        uncertainty_applied = True
    else:
        u = None
        uncertainty_applied = False

    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    mu = c * f + (1.0 - c) * p

    return {
        "name": "forecast_expected_returns",
        "method": "confidence_shrink_to_prior",
        "mu": mu.tolist(),
        "vector": mu.tolist(),
        "shape": [n],
        "n_obs": n,
        "forecasts": f.tolist(),
        "prior": p.tolist(),
        "confidence": c.tolist(),
        "uncertainty": u.tolist() if u is not None else None,
        "uncertainty_applied": uncertainty_applied,
        "prior_method": prior_method,
        "confidence_method": confidence_method,
        "names": list(names) if names is not None else None,
        "version": version,
    }
