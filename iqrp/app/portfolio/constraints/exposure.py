"""Gross / net / long / short exposure constraints."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.portfolio.constraints._types import (
    ConstraintSeverity,
    ConstraintViolation,
    as_weights,
    make_violation,
)


def exposure_metrics(weights: Any) -> dict[str, float]:
    w = as_weights(weights)
    if w.size == 0:
        return {"gross": 0.0, "net": 0.0, "long": 0.0, "short": 0.0}
    long = float(np.sum(w[w > 0.0]))
    short = float(np.sum(np.abs(w[w < 0.0])))
    return {
        "gross": float(np.sum(np.abs(w))),
        "net": float(np.sum(w)),
        "long": long,
        "short": short,
    }


def check_exposure_constraints(
    weights: Any,
    *,
    max_gross: float | None = None,
    max_net: float | None = None,
    min_net: float | None = None,
    max_long: float | None = None,
    max_short: float | None = None,
    severity: ConstraintSeverity | str = ConstraintSeverity.HARD,
) -> list[ConstraintViolation]:
    """Evaluate exposure caps. Hard violations are reported only — never auto-relaxed."""
    m = exposure_metrics(weights)
    out: list[ConstraintViolation] = []
    sev = severity

    if max_gross is not None and m["gross"] > float(max_gross) + 1e-12:
        out.append(
            make_violation(
                "max_gross_exposure",
                observed=m["gross"],
                threshold=float(max_gross),
                severity=sev,
                reason=f"gross exposure {m['gross']:.6g} exceeds max_gross {float(max_gross):.6g}",
                scope="portfolio",
            )
        )
    if max_net is not None and abs(m["net"]) > float(max_net) + 1e-12:
        out.append(
            make_violation(
                "max_net_exposure",
                observed=abs(m["net"]),
                threshold=float(max_net),
                severity=sev,
                reason=f"|net| exposure {abs(m['net']):.6g} exceeds max_net {float(max_net):.6g}",
                scope="portfolio",
                metadata={"net": m["net"]},
            )
        )
    if min_net is not None and m["net"] < float(min_net) - 1e-12:
        out.append(
            make_violation(
                "min_net_exposure",
                observed=m["net"],
                threshold=float(min_net),
                severity=sev,
                reason=f"net exposure {m['net']:.6g} below min_net {float(min_net):.6g}",
                scope="portfolio",
            )
        )
    if max_long is not None and m["long"] > float(max_long) + 1e-12:
        out.append(
            make_violation(
                "max_long_exposure",
                observed=m["long"],
                threshold=float(max_long),
                severity=sev,
                reason=f"long exposure {m['long']:.6g} exceeds max_long {float(max_long):.6g}",
                scope="portfolio",
            )
        )
    if max_short is not None and m["short"] > float(max_short) + 1e-12:
        out.append(
            make_violation(
                "max_short_exposure",
                observed=m["short"],
                threshold=float(max_short),
                severity=sev,
                reason=f"short exposure {m['short']:.6g} exceeds max_short {float(max_short):.6g}",
                scope="portfolio",
            )
        )
    return out
