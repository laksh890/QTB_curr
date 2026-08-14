"""Shared constraint types and helpers aligned with ``portfolio.base.constraints``."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from iqrp.app.portfolio.base.constraints import ConstraintViolation


class ConstraintSeverity(str, Enum):
    WARNING = "warning"
    SOFT = "soft"
    HARD = "hard"


def as_weights(weights: Any, n: int | None = None) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if n is not None and arr.size != n:
        if arr.size == 1:
            arr = np.full(n, float(arr[0]) / max(n, 1))
        else:
            out = np.zeros(n, dtype=np.float64)
            m = min(n, arr.size)
            out[:m] = arr[:m]
            arr = out
    return arr


def coerce_severity(
    severity: ConstraintSeverity | str | None,
    *,
    hard: bool | None = None,
) -> ConstraintSeverity:
    if hard is False:
        return ConstraintSeverity.SOFT
    if hard is True:
        return ConstraintSeverity.HARD
    if severity is None:
        return ConstraintSeverity.HARD
    if isinstance(severity, ConstraintSeverity):
        return severity
    s = str(severity).strip().lower()
    if s == "soft":
        return ConstraintSeverity.SOFT
    if s == "warning":
        return ConstraintSeverity.WARNING
    return ConstraintSeverity.HARD


def make_violation(
    name: str,
    *,
    observed: float,
    threshold: float,
    reason: str,
    severity: ConstraintSeverity | str = ConstraintSeverity.HARD,
    scope: str = "portfolio",
    hard: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> ConstraintViolation:
    """Build a base-compatible ``ConstraintViolation``.

    Hard constraints are flagged with ``hard=True`` and must never be auto-relaxed
    by callers of ``check_all_constraints``.
    """
    sev = coerce_severity(severity, hard=hard)
    is_hard = sev == ConstraintSeverity.HARD
    meta = dict(metadata or {})
    asset = None
    if "asset" in meta:
        asset = str(meta["asset"])
    elif "index" in meta:
        asset = str(meta["index"])
    # Preserve scope / extra metadata in message-adjacent fields via name params
    kind = str(meta.get("kind", name))
    msg = reason
    if scope and scope != "portfolio":
        msg = f"[{scope}] {reason}"
    return ConstraintViolation(
        name=name,
        kind=kind,
        actual=float(observed),
        limit=float(threshold),
        message=msg,
        hard=is_hard,
        asset=asset,
    )


def filter_by_severity(
    violations: list[ConstraintViolation],
    *,
    include_soft: bool = True,
    include_hard: bool = True,
) -> list[ConstraintViolation]:
    out: list[ConstraintViolation] = []
    for v in violations:
        is_hard = bool(getattr(v, "hard", True))
        if (is_hard and include_hard) or ((not is_hard) and include_soft):
            out.append(v)
    return out
