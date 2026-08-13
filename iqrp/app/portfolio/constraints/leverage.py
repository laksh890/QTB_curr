"""Leverage constraints (gross leverage and signed leverage)."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.portfolio.constraints._types import (
    ConstraintSeverity,
    ConstraintViolation,
    as_weights,
    make_violation,
)


def leverage_metrics(weights: Any) -> dict[str, float]:
    w = as_weights(weights)
    gross = float(np.sum(np.abs(w))) if w.size else 0.0
    net = float(np.sum(w)) if w.size else 0.0
    return {
        "leverage": gross,
        "gross_leverage": gross,
        "net_leverage": net,
    }


def check_leverage_constraints(
    weights: Any,
    *,
    max_leverage: float | None = None,
    min_leverage: float | None = None,
    severity: ConstraintSeverity | str = ConstraintSeverity.HARD,
) -> list[ConstraintViolation]:
    """Check leverage bounds. Hard limits are never auto-relaxed."""
    m = leverage_metrics(weights)
    out: list[ConstraintViolation] = []
    lev = m["leverage"]

    if max_leverage is not None and lev > float(max_leverage) + 1e-12:
        out.append(
            make_violation(
                "max_leverage",
                observed=lev,
                threshold=float(max_leverage),
                severity=severity,
                reason=f"leverage {lev:.6g} exceeds max_leverage {float(max_leverage):.6g}",
            )
        )
    if min_leverage is not None and lev < float(min_leverage) - 1e-12:
        out.append(
            make_violation(
                "min_leverage",
                observed=lev,
                threshold=float(min_leverage),
                severity=severity,
                reason=f"leverage {lev:.6g} below min_leverage {float(min_leverage):.6g}",
            )
        )
    return out
