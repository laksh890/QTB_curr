"""Liquidity limits."""

from __future__ import annotations

from iqrp.app.risk.base import LimitBreach, LimitSeverity, RiskLimit, evaluate_limits


def build_liquidity_limits(
    *,
    max_participation: float = 0.10,
    min_adv_coverage: float = 0.01,
    max_time_to_liquidate: float = 5.0,
    severity: LimitSeverity = LimitSeverity.HARD,
) -> list[RiskLimit]:
    return [
        RiskLimit(
            name="max_participation",
            threshold=float(max_participation),
            severity=severity,
            scope="position",
            direction="max",
        ),
        RiskLimit(
            name="min_adv_coverage",
            threshold=float(min_adv_coverage),
            severity=severity,
            scope="position",
            direction="min",
            metadata={"description": "ADV notional / position must stay above threshold"},
        ),
        RiskLimit(
            name="max_time_to_liquidate",
            threshold=float(max_time_to_liquidate),
            severity=LimitSeverity.SOFT if severity == LimitSeverity.HARD else severity,
            scope="position",
            direction="max",
            metadata={"unit": "days"},
        ),
    ]


def check_liquidity_limits(
    *,
    participation: float,
    adv_coverage: float,
    time_to_liquidate: float = 0.0,
    limits: list[RiskLimit] | None = None,
    max_participation: float = 0.10,
    min_adv_coverage: float = 0.01,
    max_time_to_liquidate: float = 5.0,
) -> list[LimitBreach]:
    lims = limits if limits is not None else build_liquidity_limits(
        max_participation=max_participation,
        min_adv_coverage=min_adv_coverage,
        max_time_to_liquidate=max_time_to_liquidate,
    )
    values = {
        "max_participation": float(participation),
        "min_adv_coverage": float(adv_coverage),
        "max_time_to_liquidate": float(time_to_liquidate),
    }
    return evaluate_limits(lims, values)
