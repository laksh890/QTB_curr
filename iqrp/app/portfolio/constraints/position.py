"""Per-position weight / box constraints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from iqrp.app.portfolio.constraints._types import (
    ConstraintSeverity,
    ConstraintViolation,
    as_weights,
    make_violation,
)


def check_position_constraints(
    weights: Any,
    *,
    max_position: float | Sequence[float] | None = None,
    min_position: float | Sequence[float] | None = None,
    max_weight: float | Sequence[float] | None = None,
    min_weight: float | Sequence[float] | None = None,
    long_only: bool = False,
    severity: ConstraintSeverity | str = ConstraintSeverity.HARD,
) -> list[ConstraintViolation]:
    """Box constraints on individual weights. Hard bounds are never auto-relaxed."""
    w = as_weights(weights)
    n = int(w.size)
    if n == 0:
        return []

    hi_spec = max_position if max_position is not None else max_weight
    lo_spec = min_position if min_position is not None else min_weight

    def _bound_vec(spec: float | Sequence[float] | None) -> np.ndarray | None:
        if spec is None:
            return None
        arr = np.asarray(spec, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            return np.full(n, float(arr[0]))
        out = np.zeros(n, dtype=np.float64)
        m = min(n, arr.size)
        out[:m] = arr[:m]
        if m < n:
            out[m:] = float(arr[-1])
        return out

    hi = _bound_vec(hi_spec)
    lo = _bound_vec(lo_spec)
    out: list[ConstraintViolation] = []

    for i, wi in enumerate(w):
        val = float(wi)
        if long_only and val < -1e-12:
            out.append(
                make_violation(
                    "long_only",
                    observed=val,
                    threshold=0.0,
                    severity=severity,
                    reason=f"weight[{i}]={val:.6g} violates long_only",
                    scope="position",
                    metadata={"index": int(i)},
                )
            )
        if hi is not None and abs(val) > float(hi[i]) + 1e-12:
            out.append(
                make_violation(
                    "max_position",
                    observed=abs(val),
                    threshold=float(hi[i]),
                    severity=severity,
                    reason=(
                        f"|weight[{i}]|={abs(val):.6g} exceeds " f"max_position {float(hi[i]):.6g}"
                    ),
                    scope="position",
                    metadata={"index": int(i), "weight": val},
                )
            )
        if lo is not None and val < float(lo[i]) - 1e-12:
            out.append(
                make_violation(
                    "min_position",
                    observed=val,
                    threshold=float(lo[i]),
                    severity=severity,
                    reason=f"weight[{i}]={val:.6g} below min_position {float(lo[i]):.6g}",
                    scope="position",
                    metadata={"index": int(i)},
                )
            )
    return out
