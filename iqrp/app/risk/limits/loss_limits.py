"""Loss and drawdown limits."""

from __future__ import annotations

from iqrp.app.risk.base import LimitBreach, LimitSeverity, RiskLimit, evaluate_limits


def build_loss_limits(
    *,
    max_daily_loss: float = 0.03,
    max_drawdown: float = 0.20,
    max_weekly_loss: float | None = None,
    severity: LimitSeverity = LimitSeverity.HARD,
) -> list[RiskLimit]:
    limits = [
        RiskLimit(
            name="max_daily_loss",
            threshold=float(max_daily_loss),
            severity=severity,
            scope="portfolio",
            direction="max",
            metadata={"description": "Absolute daily P&L loss fraction"},
        ),
        RiskLimit(
            name="max_drawdown",
            threshold=float(max_drawdown),
            severity=LimitSeverity.HARD,  # drawdown always hard
            scope="portfolio",
            direction="max",
            metadata={"description": "Hard drawdown stop — not overridable by confidence"},
        ),
    ]
    if max_weekly_loss is not None:
        limits.append(
            RiskLimit(
                name="max_weekly_loss",
                threshold=float(max_weekly_loss),
                severity=severity,
                scope="portfolio",
                direction="max",
            )
        )
    return limits


def check_loss_limits(
    *,
    daily_loss: float = 0.0,
    current_drawdown: float = 0.0,
    weekly_loss: float | None = None,
    limits: list[RiskLimit] | None = None,
    max_daily_loss: float = 0.03,
    max_drawdown: float = 0.20,
) -> list[LimitBreach]:
    lims = (
        limits
        if limits is not None
        else build_loss_limits(
            max_daily_loss=max_daily_loss,
            max_drawdown=max_drawdown,
        )
    )
    values = {
        "max_daily_loss": abs(float(daily_loss)),
        "max_drawdown": abs(float(current_drawdown)),
    }
    if weekly_loss is not None:
        values["max_weekly_loss"] = abs(float(weekly_loss))
    return evaluate_limits(lims, values)
