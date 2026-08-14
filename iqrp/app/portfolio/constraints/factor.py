"""Factor exposure constraints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from iqrp.app.portfolio.constraints._types import (
    ConstraintSeverity,
    ConstraintViolation,
    as_weights,
    make_violation,
)


def portfolio_factor_exposures(
    weights: Any,
    factor_loadings: Any,
    *,
    factor_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Portfolio factor exposure = B' w (loadings shape n x k or k x n)."""
    w = as_weights(weights)
    B = np.asarray(factor_loadings, dtype=np.float64)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    if B.ndim != 2:
        raise ValueError("factor_loadings must be 1-D or 2-D")
    if B.shape[0] == w.size:
        pass
    elif B.shape[1] == w.size:
        B = B.T
    else:
        n = w.size
        # pad / truncate rows
        out = np.zeros((n, B.shape[1] if B.shape[0] != n else B.shape[1]), dtype=np.float64)
        if B.shape[0] >= n:
            out = B[:n, :]
        else:
            out[: B.shape[0], : B.shape[1]] = B
        B = out
    expo = B.T @ w
    k = int(expo.size)
    names = list(factor_names) if factor_names is not None else [f"factor_{i}" for i in range(k)]
    if len(names) < k:
        names = names + [f"factor_{i}" for i in range(len(names), k)]
    return {names[i]: float(expo[i]) for i in range(k)}


def check_factor_constraints(
    weights: Any,
    *,
    factor_loadings: Any | None = None,
    factor_names: Sequence[str] | None = None,
    max_factor_exposure: float | Mapping[str, float] | None = None,
    min_factor_exposure: float | Mapping[str, float] | None = None,
    factor_neutral: bool | Sequence[str] = False,
    neutrality_tol: float = 1e-6,
    severity: ConstraintSeverity | str = ConstraintSeverity.HARD,
) -> list[ConstraintViolation]:
    if factor_loadings is None:
        return []
    if max_factor_exposure is None and min_factor_exposure is None and not factor_neutral:
        return []

    exposures = portfolio_factor_exposures(weights, factor_loadings, factor_names=factor_names)
    out: list[ConstraintViolation] = []

    neutral_set: set[str]
    if factor_neutral is True:
        neutral_set = set(exposures)
    elif isinstance(factor_neutral, (list, tuple, set)):
        neutral_set = {str(x) for x in factor_neutral}
    else:
        neutral_set = set()

    def _bound(spec: float | Mapping[str, float] | None, name: str) -> float | None:
        if spec is None:
            return None
        if isinstance(spec, Mapping):
            return float(spec[name]) if name in spec else None
        return float(spec)

    for name, exp in exposures.items():
        abs_exp = abs(float(exp))
        if name in neutral_set and abs_exp > float(neutrality_tol):
            out.append(
                make_violation(
                    "factor_neutrality",
                    observed=abs_exp,
                    threshold=float(neutrality_tol),
                    severity=severity,
                    reason=f"factor '{name}' exposure={float(exp):.6g} not neutral",
                    metadata={"factor": name, "exposure": float(exp)},
                )
            )
        hi = _bound(max_factor_exposure, name)
        lo = _bound(min_factor_exposure, name)
        if hi is not None and abs_exp > hi + 1e-12:
            out.append(
                make_violation(
                    "max_factor_exposure",
                    observed=abs_exp,
                    threshold=hi,
                    severity=severity,
                    reason=f"factor '{name}' |exposure|={abs_exp:.6g} exceeds {hi:.6g}",
                    metadata={"factor": name, "exposure": float(exp)},
                )
            )
        if lo is not None and float(exp) < lo - 1e-12:
            out.append(
                make_violation(
                    "min_factor_exposure",
                    observed=float(exp),
                    threshold=lo,
                    severity=severity,
                    reason=f"factor '{name}' exposure={float(exp):.6g} below {lo:.6g}",
                    metadata={"factor": name},
                )
            )
    return out
