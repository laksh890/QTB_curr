"""Concentration limits."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.base import LimitBreach, LimitSeverity, RiskLimit, as_weights, evaluate_limits
from iqrp.app.risk.portfolio.concentration import herfindahl, max_weight


def build_concentration_limits(
    *,
    max_concentration: float = 0.25,
    max_herfindahl: float = 0.30,
    severity: LimitSeverity = LimitSeverity.HARD,
) -> list[RiskLimit]:
    return [
        RiskLimit(
            name="max_concentration",
            threshold=float(max_concentration),
            severity=severity,
            scope="portfolio",
            direction="max",
            metadata={"description": "Max absolute single weight"},
        ),
        RiskLimit(
            name="max_herfindahl",
            threshold=float(max_herfindahl),
            severity=LimitSeverity.SOFT if severity == LimitSeverity.HARD else severity,
            scope="portfolio",
            direction="max",
        ),
    ]


def check_concentration_limits(
    weights: Any,
    *,
    limits: list[RiskLimit] | None = None,
    max_concentration: float = 0.25,
    max_herfindahl: float = 0.30,
) -> list[LimitBreach]:
    lims = (
        limits
        if limits is not None
        else build_concentration_limits(
            max_concentration=max_concentration,
            max_herfindahl=max_herfindahl,
        )
    )
    w = as_weights(weights)
    values = {
        "max_concentration": max_weight(w).value,
        "max_herfindahl": herfindahl(w).value,
    }
    return evaluate_limits(lims, values)
