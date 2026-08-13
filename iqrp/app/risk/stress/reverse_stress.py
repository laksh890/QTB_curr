"""Reverse stress: find shock magnitude that breaches a loss limit."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_weights


def reverse_stress(
    weights: Any,
    direction: Any,
    *,
    loss_limit: float,
    max_magnitude: float = 5.0,
    n_grid: int = 200,
) -> dict[str, Any]:
    """Find the smallest magnitude m such that -w · (m * u) >= loss_limit.

    ``direction`` is a shock direction vector (normalized internally).
    Uses a deterministic grid search (no randomness, no future data).
    """
    w = as_weights(weights)
    u = np.asarray(direction, dtype=np.float64).reshape(-1)
    if u.size != w.size:
        tmp = np.zeros(w.size, dtype=np.float64)
        m = min(w.size, u.size)
        tmp[:m] = u[:m]
        u = tmp
    norm = float(np.linalg.norm(u))
    if norm <= 1e-12:
        u = np.ones(w.size, dtype=np.float64) / max(np.sqrt(w.size), 1.0)
    else:
        u = u / norm

    limit = abs(float(loss_limit))
    # PnL(m) = w · (m * u) = m * (w·u); loss = max(-PnL, 0)
    exposure = float(np.dot(w, u))
    if abs(exposure) <= 1e-15:
        # Direction orthogonal to weights — cannot breach via this direction
        return {
            "name": "reverse_stress",
            "breach_possible": False,
            "magnitude": None,
            "loss_limit": limit,
            "direction_exposure": exposure,
            "measures": {
                "reverse_stress_magnitude": RiskMeasure(
                    name="reverse_stress_magnitude",
                    value=float("inf"),
                    unit="shock",
                    method="reverse_stress",
                    parameters={"loss_limit": limit, "breach_possible": False},
                ).to_dict(),
            },
        }

    # Loss(m) = max(-m * exposure, 0). Need -m * exposure >= limit
    # If exposure < 0, positive m increases loss: m >= limit / (-exposure)
    # If exposure > 0, need negative m: |m| >= limit / exposure
    if exposure < 0:
        analytic = limit / (-exposure)
        sign = 1.0
    else:
        analytic = limit / exposure
        sign = -1.0

    # Grid refine for numerical reporting
    grid = np.linspace(0.0, max(float(max_magnitude), analytic * 1.1), max(int(n_grid), 10))
    found = None
    for m in grid:
        shock = sign * m * u
        pnl = float(np.dot(w, shock))
        loss = max(-pnl, 0.0)
        if loss >= limit - 1e-12:
            found = float(m)
            break

    magnitude = float(min(analytic, found if found is not None else analytic))
    if magnitude > float(max_magnitude):
        breach_possible = False
        magnitude_out: float | None = None
    else:
        breach_possible = True
        magnitude_out = magnitude

    return {
        "name": "reverse_stress",
        "breach_possible": breach_possible,
        "magnitude": magnitude_out,
        "signed_magnitude": float(sign * magnitude) if breach_possible else None,
        "loss_limit": limit,
        "direction_exposure": exposure,
        "unit_direction": u.tolist(),
        "measures": {
            "reverse_stress_magnitude": RiskMeasure(
                name="reverse_stress_magnitude",
                value=float(magnitude_out) if magnitude_out is not None else float("inf"),
                unit="shock",
                method="reverse_stress",
                parameters={
                    "loss_limit": limit,
                    "breach_possible": breach_possible,
                    "max_magnitude": float(max_magnitude),
                },
            ).to_dict(),
        },
    }
