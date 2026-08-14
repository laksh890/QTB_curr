"""Turnover constraints and no-trade region checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.portfolio.constraints._types import (
    ConstraintSeverity,
    ConstraintViolation,
    as_weights,
    make_violation,
)


def turnover(weights_old: Any, weights_new: Any) -> float:
    """One-way turnover = 0.5 * L1(weight delta)."""
    a = as_weights(weights_old)
    b = as_weights(weights_new)
    n = max(a.size, b.size)
    a = as_weights(a, n=n)
    b = as_weights(b, n=n)
    return 0.5 * float(np.sum(np.abs(b - a)))


def check_turnover_constraints(
    weights: Any,
    *,
    weights_old: Any | None = None,
    current_weights: Any | None = None,
    max_turnover: float | None = None,
    min_trade: float | None = None,
    severity: ConstraintSeverity | str = ConstraintSeverity.HARD,
) -> list[ConstraintViolation]:
    """Hard turnover caps are reported only — never silently scaled."""
    prev = weights_old if weights_old is not None else current_weights
    if prev is None or (max_turnover is None and min_trade is None):
        return []

    new = as_weights(weights)
    old = as_weights(prev)
    n = max(new.size, old.size)
    new = as_weights(new, n=n)
    old = as_weights(old, n=n)
    delta = new - old
    to = 0.5 * float(np.sum(np.abs(delta)))
    out: list[ConstraintViolation] = []

    if max_turnover is not None and to > float(max_turnover) + 1e-12:
        out.append(
            make_violation(
                "max_turnover",
                observed=to,
                threshold=float(max_turnover),
                severity=severity,
                reason=f"turnover {to:.6g} exceeds max_turnover {float(max_turnover):.6g}",
                metadata={"one_way": True},
            )
        )

    if min_trade is not None:
        thr = float(min_trade)
        # Flag tiny non-zero trades that fall in the no-trade dead zone (soft by nature of check)
        for i, d in enumerate(delta):
            ad = float(abs(d))
            if 0.0 < ad + 0.0 < thr - 1e-15 and ad > 1e-15:
                out.append(
                    make_violation(
                        "min_trade",
                        observed=ad,
                        threshold=thr,
                        severity=(
                            ConstraintSeverity.SOFT
                            if str(getattr(severity, "value", severity)).lower() != "hard"
                            else severity
                        ),
                        reason=(
                            f"|delta_w[{i}]|={ad:.6g} is below min_trade {thr:.6g} "
                            "(no-trade region)"
                        ),
                        scope="position",
                        metadata={"index": int(i), "delta": float(d)},
                    )
                )
    return out
