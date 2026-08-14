"""Currency exposure constraints."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from iqrp.app.portfolio.constraints._types import (
    ConstraintSeverity,
    ConstraintViolation,
    as_weights,
    make_violation,
)


def currency_exposures(
    weights: Any,
    currencies: Sequence[str] | Mapping[int, str] | None,
) -> dict[str, float]:
    w = as_weights(weights)
    if currencies is None or w.size == 0:
        return {}
    out: dict[str, float] = defaultdict(float)
    for i, wi in enumerate(w):
        if isinstance(currencies, Mapping):
            ccy = str(currencies.get(i, currencies.get(str(i), "USD")))  # type: ignore[arg-type]
        else:
            ccy = str(currencies[i]) if i < len(currencies) else "USD"
        out[ccy] += float(wi)
    return dict(out)


def check_currency_constraints(
    weights: Any,
    *,
    currencies: Sequence[str] | Mapping[int, str] | None = None,
    max_currency_exposure: float | Mapping[str, float] | None = None,
    min_currency_exposure: float | Mapping[str, float] | None = None,
    severity: ConstraintSeverity | str = ConstraintSeverity.HARD,
) -> list[ConstraintViolation]:
    if currencies is None or (max_currency_exposure is None and min_currency_exposure is None):
        return []
    exposures = currency_exposures(weights, currencies)
    out: list[ConstraintViolation] = []

    def _bound(spec: float | Mapping[str, float] | None, ccy: str) -> float | None:
        if spec is None:
            return None
        if isinstance(spec, Mapping):
            return float(spec[ccy]) if ccy in spec else None
        return float(spec)

    for ccy, exp in exposures.items():
        abs_exp = abs(float(exp))
        hi = _bound(max_currency_exposure, ccy)
        lo = _bound(min_currency_exposure, ccy)
        if hi is not None and abs_exp > hi + 1e-12:
            out.append(
                make_violation(
                    "max_currency_exposure",
                    observed=abs_exp,
                    threshold=hi,
                    severity=severity,
                    reason=f"currency '{ccy}' |exposure|={abs_exp:.6g} exceeds {hi:.6g}",
                    metadata={"currency": ccy, "exposure": float(exp)},
                )
            )
        if lo is not None and float(exp) < lo - 1e-12:
            out.append(
                make_violation(
                    "min_currency_exposure",
                    observed=float(exp),
                    threshold=lo,
                    severity=severity,
                    reason=f"currency '{ccy}' exposure={float(exp):.6g} below {lo:.6g}",
                    metadata={"currency": ccy},
                )
            )
    return out
