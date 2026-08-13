"""Concentration constraints: max weight, HHI, effective N."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.portfolio.constraints._types import (
    ConstraintSeverity,
    ConstraintViolation,
    as_weights,
    make_violation,
)


def concentration_metrics(weights: Any) -> dict[str, float]:
    w = as_weights(weights)
    if w.size == 0:
        return {"max_weight": 0.0, "hhi": 0.0, "effective_n": 0.0}
    abs_w = np.abs(w)
    total = float(np.sum(abs_w))
    shares = abs_w / total if total > 0 else np.zeros_like(abs_w)
    hhi = float(np.sum(shares**2))
    effective_n = float(1.0 / hhi) if hhi > 1e-18 else 0.0
    return {
        "max_weight": float(np.max(abs_w)),
        "hhi": hhi,
        "effective_n": effective_n,
        "n_assets": float(w.size),
    }


def check_concentration_constraints(
    weights: Any,
    *,
    max_weight: float | None = None,
    max_hhi: float | None = None,
    min_effective_n: float | None = None,
    severity: ConstraintSeverity | str = ConstraintSeverity.HARD,
) -> list[ConstraintViolation]:
    """Max weight / HHI / effective-N constraints. Never auto-relax hard caps."""
    m = concentration_metrics(weights)
    w = as_weights(weights)
    out: list[ConstraintViolation] = []

    if max_weight is not None:
        thr = float(max_weight)
        for i, wi in enumerate(w):
            obs = float(abs(wi))
            if obs > thr + 1e-12:
                out.append(
                    make_violation(
                        "max_weight",
                        observed=obs,
                        threshold=thr,
                        severity=severity,
                        reason=f"abs(weight[{i}])={obs:.6g} exceeds max_weight {thr:.6g}",
                        scope="position",
                        metadata={"index": int(i), "weight": float(wi)},
                    )
                )

    if max_hhi is not None and m["hhi"] > float(max_hhi) + 1e-12:
        out.append(
            make_violation(
                "max_hhi",
                observed=m["hhi"],
                threshold=float(max_hhi),
                severity=severity,
                reason=f"HHI {m['hhi']:.6g} exceeds max_hhi {float(max_hhi):.6g}",
                metadata={"effective_n": m["effective_n"]},
            )
        )

    if min_effective_n is not None and m["effective_n"] < float(min_effective_n) - 1e-12:
        out.append(
            make_violation(
                "min_effective_n",
                observed=m["effective_n"],
                threshold=float(min_effective_n),
                severity=severity,
                reason=(
                    f"effective N {m['effective_n']:.6g} below "
                    f"min_effective_n {float(min_effective_n):.6g}"
                ),
                metadata={"hhi": m["hhi"]},
            )
        )
    return out
