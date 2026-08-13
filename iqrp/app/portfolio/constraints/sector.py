"""Sector exposure constraints."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from iqrp.app.portfolio.constraints._types import (
    ConstraintSeverity,
    ConstraintViolation,
    as_weights,
    make_violation,
)


def sector_exposures(
    weights: Any,
    sector_map: Sequence[str] | Mapping[int, str] | Mapping[str, str] | None,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Aggregate signed weight by sector label."""
    w = as_weights(weights)
    if sector_map is None or w.size == 0:
        return {}
    out: dict[str, float] = defaultdict(float)
    for i, wi in enumerate(w):
        if isinstance(sector_map, Mapping):
            if names is not None and i < len(names) and names[i] in sector_map:
                sec = str(sector_map[names[i]])  # type: ignore[index]
            elif i in sector_map:
                sec = str(sector_map[i])  # type: ignore[index]
            elif str(i) in sector_map:
                sec = str(sector_map[str(i)])  # type: ignore[index]
            else:
                sec = "UNKNOWN"
        else:
            sec = str(sector_map[i]) if i < len(sector_map) else "UNKNOWN"
        out[sec] += float(wi)
    return dict(out)


def check_sector_constraints(
    weights: Any,
    *,
    sector_map: Sequence[str] | Mapping[int, str] | Mapping[str, str] | None = None,
    names: Sequence[str] | None = None,
    max_sector_weight: float | Mapping[str, float] | None = None,
    min_sector_weight: float | Mapping[str, float] | None = None,
    severity: ConstraintSeverity | str = ConstraintSeverity.HARD,
) -> list[ConstraintViolation]:
    if sector_map is None or (max_sector_weight is None and min_sector_weight is None):
        return []
    exposures = sector_exposures(weights, sector_map, names=names)
    out: list[ConstraintViolation] = []

    def _cap(spec: float | Mapping[str, float] | None, sector: str) -> float | None:
        if spec is None:
            return None
        if isinstance(spec, Mapping):
            return float(spec[sector]) if sector in spec else None
        return float(spec)

    for sec, exp in exposures.items():
        abs_exp = abs(float(exp))
        hi = _cap(max_sector_weight, sec)
        lo = _cap(min_sector_weight, sec)
        if hi is not None and abs_exp > hi + 1e-12:
            out.append(
                make_violation(
                    "max_sector_weight",
                    observed=abs_exp,
                    threshold=hi,
                    severity=severity,
                    reason=f"sector '{sec}' |exposure|={abs_exp:.6g} exceeds {hi:.6g}",
                    metadata={"sector": sec, "exposure": float(exp)},
                )
            )
        if lo is not None and float(exp) < lo - 1e-12:
            out.append(
                make_violation(
                    "min_sector_weight",
                    observed=float(exp),
                    threshold=lo,
                    severity=severity,
                    reason=f"sector '{sec}' exposure={float(exp):.6g} below {lo:.6g}",
                    metadata={"sector": sec},
                )
            )
    return out
