"""Thin wrapper re-exporting robust portfolio optimizers."""

from __future__ import annotations

from typing import Any

from iqrp.app.portfolio.robust.distributional_robust import optimize_distributional_robust
from iqrp.app.portfolio.robust.parameter_uncertainty import (
    optimize_parameter_uncertainty,
    optimize_robust_mean_variance,
)


def optimize_robust(
    mu: Any = None,
    cov: Any = None,
    *,
    current_weights: Any = None,
    constraints: Any = None,
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_aversion: float = 1.0,
    mode: str = "distributional",
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Robust optimizer entrypoint.

    mode:
      - distributional / ellipsoidal / box → worst-case mu then MV
      - parameter / se → estimation-error uncertainty set
    """
    m = str(mode).lower()
    if m in {"parameter", "parameter_uncertainty", "se", "estimation"}:
        return optimize_parameter_uncertainty(
            mu=mu,
            cov=cov,
            current_weights=current_weights,
            constraints=constraints,
            long_only=long_only,
            max_weight=max_weight,
            risk_aversion=risk_aversion,
            **kwargs,
        )
    # map mode to uncertainty kind when string-like
    if "uncertainty" not in kwargs:
        if m in {"box", "interval"}:
            kwargs["uncertainty"] = "box"
        elif m in {"ellipsoidal", "ellipsoid"}:
            kwargs["uncertainty"] = "ellipsoidal"
    res = optimize_distributional_robust(
        mu=mu,
        cov=cov,
        current_weights=current_weights,
        constraints=constraints,
        long_only=long_only,
        max_weight=max_weight,
        risk_aversion=risk_aversion,
        **kwargs,
    )
    out = dict(res)
    out["name"] = "robust"
    return out


__all__ = [
    "optimize_distributional_robust",
    "optimize_parameter_uncertainty",
    "optimize_robust",
    "optimize_robust_mean_variance",
]
