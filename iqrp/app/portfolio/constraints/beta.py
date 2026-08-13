"""Portfolio beta constraints."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.portfolio.constraints._types import (
    ConstraintSeverity,
    ConstraintViolation,
    as_weights,
    make_violation,
)


def portfolio_beta(weights: Any, betas: Any) -> float:
    w = as_weights(weights)
    b = np.asarray(betas, dtype=np.float64).reshape(-1)
    n = max(w.size, b.size)
    w = as_weights(w, n=n)
    if b.size == 1:
        b = np.full(n, float(b[0]))
    elif b.size != n:
        bb = np.zeros(n, dtype=np.float64)
        m = min(n, b.size)
        bb[:m] = b[:m]
        b = bb
    return float(w @ b)


def check_beta_constraints(
    weights: Any,
    *,
    betas: Any | None = None,
    portfolio_beta_value: float | None = None,
    max_beta: float | None = None,
    min_beta: float | None = None,
    target_beta: float | None = None,
    beta_tol: float = 0.05,
    severity: ConstraintSeverity | str = ConstraintSeverity.HARD,
) -> list[ConstraintViolation]:
    if betas is None and portfolio_beta_value is None:
        return []
    if max_beta is None and min_beta is None and target_beta is None:
        return []

    beta_val = (
        float(portfolio_beta_value)
        if portfolio_beta_value is not None
        else portfolio_beta(weights, betas)
    )
    out: list[ConstraintViolation] = []

    if max_beta is not None and beta_val > float(max_beta) + 1e-12:
        out.append(
            make_violation(
                "max_beta",
                observed=beta_val,
                threshold=float(max_beta),
                severity=severity,
                reason=f"portfolio beta {beta_val:.6g} exceeds max_beta {float(max_beta):.6g}",
            )
        )
    if min_beta is not None and beta_val < float(min_beta) - 1e-12:
        out.append(
            make_violation(
                "min_beta",
                observed=beta_val,
                threshold=float(min_beta),
                severity=severity,
                reason=f"portfolio beta {beta_val:.6g} below min_beta {float(min_beta):.6g}",
            )
        )
    if target_beta is not None and abs(beta_val - float(target_beta)) > float(beta_tol) + 1e-12:
        out.append(
            make_violation(
                "target_beta",
                observed=beta_val,
                threshold=float(target_beta),
                severity=severity,
                reason=(
                    f"portfolio beta {beta_val:.6g} outside target "
                    f"{float(target_beta):.6g} ± {float(beta_tol):.6g}"
                ),
                metadata={"beta_tol": float(beta_tol)},
            )
        )
    return out
