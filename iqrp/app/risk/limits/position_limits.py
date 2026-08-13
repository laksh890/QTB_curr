"""Position-level risk limits."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.base import LimitBreach, LimitSeverity, RiskLimit, evaluate_limits


def build_position_limits(
    *,
    max_position: float = 0.10,
    max_single_name: float | None = None,
    severity: LimitSeverity = LimitSeverity.HARD,
) -> list[RiskLimit]:
    """Build position weight / notional fraction limits."""
    limits = [
        RiskLimit(
            name="max_position",
            threshold=float(max_position),
            severity=severity,
            scope="position",
            direction="max",
            metadata={"description": "Absolute position weight cap"},
        )
    ]
    if max_single_name is not None:
        limits.append(
            RiskLimit(
                name="max_single_name",
                threshold=float(max_single_name),
                severity=severity,
                scope="position",
                direction="max",
                metadata={"description": "Single-name concentration cap"},
            )
        )
    return limits


def check_position_limits(
    *,
    position_weight: float,
    limits: list[RiskLimit] | None = None,
    max_position: float = 0.10,
) -> list[LimitBreach]:
    lims = limits if limits is not None else build_position_limits(max_position=max_position)
    values: dict[str, float] = {"max_position": abs(float(position_weight))}
    # Also map max_single_name if present
    for lim in lims:
        if lim.name == "max_single_name":
            values["max_single_name"] = abs(float(position_weight))
    return evaluate_limits(lims, values)


def check_positions(
    weights: Any,
    *,
    limits: list[RiskLimit] | None = None,
    max_position: float = 0.10,
) -> list[LimitBreach]:
    """Check each weight against position limits."""
    from iqrp.app.risk.base import as_weights

    w = as_weights(weights)
    lims = limits if limits is not None else build_position_limits(max_position=max_position)
    breaches: list[LimitBreach] = []
    for i, wi in enumerate(w):
        for breach in check_position_limits(
            position_weight=float(wi), limits=lims, max_position=max_position
        ):
            meta = dict(breach.metadata)
            meta["index"] = int(i)
            breaches.append(
                LimitBreach(
                    limit_name=breach.limit_name,
                    severity=breach.severity,
                    observed=breach.observed,
                    threshold=breach.threshold,
                    reason=breach.reason,
                    scope=breach.scope,
                    metadata=meta,
                )
            )
    return breaches
