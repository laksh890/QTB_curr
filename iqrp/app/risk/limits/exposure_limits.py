"""Gross / net exposure limits."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.base import LimitBreach, LimitSeverity, RiskLimit, as_weights, evaluate_limits
from iqrp.app.risk.portfolio.exposure import gross_exposure, net_exposure


def build_exposure_limits(
    *,
    max_gross_exposure: float = 1.5,
    max_net_exposure: float = 1.0,
    severity: LimitSeverity = LimitSeverity.HARD,
) -> list[RiskLimit]:
    return [
        RiskLimit(
            name="max_gross_exposure",
            threshold=float(max_gross_exposure),
            severity=severity,
            scope="portfolio",
            direction="max",
        ),
        RiskLimit(
            name="max_net_exposure",
            threshold=float(max_net_exposure),
            severity=severity,
            scope="portfolio",
            direction="max",
            metadata={"note": "Compared against abs(net exposure)"},
        ),
    ]


def check_exposure_limits(
    weights: Any,
    *,
    limits: list[RiskLimit] | None = None,
    max_gross_exposure: float = 1.5,
    max_net_exposure: float = 1.0,
) -> list[LimitBreach]:
    lims = limits if limits is not None else build_exposure_limits(
        max_gross_exposure=max_gross_exposure,
        max_net_exposure=max_net_exposure,
    )
    w = as_weights(weights)
    values = {
        "max_gross_exposure": gross_exposure(w).value,
        "max_net_exposure": abs(net_exposure(w).value),
    }
    return evaluate_limits(lims, values)
