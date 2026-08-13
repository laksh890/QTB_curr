"""Limit builders and aggregate checker."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.base import LimitBreach, LimitSeverity, RiskLimit
from iqrp.app.risk.limits.concentration_limits import (
    build_concentration_limits,
    check_concentration_limits,
)
from iqrp.app.risk.limits.exposure_limits import build_exposure_limits, check_exposure_limits
from iqrp.app.risk.limits.liquidity_limits import build_liquidity_limits, check_liquidity_limits
from iqrp.app.risk.limits.loss_limits import build_loss_limits, check_loss_limits
from iqrp.app.risk.limits.position_limits import (
    build_position_limits,
    check_position_limits,
    check_positions,
)


def build_default_limits(
    *,
    max_position: float = 0.10,
    max_gross_exposure: float = 1.5,
    max_net_exposure: float = 1.0,
    max_concentration: float = 0.25,
    max_daily_loss: float = 0.03,
    max_drawdown: float = 0.20,
    max_participation: float = 0.10,
    min_adv_coverage: float = 0.01,
) -> list[RiskLimit]:
    return (
        build_position_limits(max_position=max_position)
        + build_exposure_limits(
            max_gross_exposure=max_gross_exposure,
            max_net_exposure=max_net_exposure,
        )
        + build_loss_limits(max_daily_loss=max_daily_loss, max_drawdown=max_drawdown)
        + build_concentration_limits(max_concentration=max_concentration)
        + build_liquidity_limits(
            max_participation=max_participation,
            min_adv_coverage=min_adv_coverage,
        )
    )


def check_all_limits(
    *,
    weights: Any | None = None,
    daily_loss: float = 0.0,
    current_drawdown: float = 0.0,
    participation: float | None = None,
    adv_coverage: float | None = None,
    time_to_liquidate: float = 0.0,
    limits: list[RiskLimit] | None = None,
    max_position: float = 0.10,
    max_gross_exposure: float = 1.5,
    max_net_exposure: float = 1.0,
    max_concentration: float = 0.25,
    max_daily_loss: float = 0.03,
    max_drawdown: float = 0.20,
    max_participation: float = 0.10,
    min_adv_coverage: float = 0.01,
) -> list[LimitBreach]:
    """Run all configured limit checks and return combined breaches.

    Hard breaches (e.g. drawdown) are never softened by forecast confidence —
    this function does not accept a confidence override parameter.
    """
    breaches: list[LimitBreach] = []

    pos_lims = [L for L in (limits or []) if L.name in ("max_position", "max_single_name")]
    exp_lims = [L for L in (limits or []) if L.name in ("max_gross_exposure", "max_net_exposure")]
    loss_lims = [L for L in (limits or []) if L.name in ("max_daily_loss", "max_drawdown", "max_weekly_loss")]
    conc_lims = [L for L in (limits or []) if L.name in ("max_concentration", "max_herfindahl")]
    liq_lims = [
        L
        for L in (limits or [])
        if L.name in ("max_participation", "min_adv_coverage", "max_time_to_liquidate")
    ]

    if weights is not None:
        breaches.extend(
            check_positions(
                weights,
                limits=pos_lims or None,
                max_position=max_position,
            )
        )
        breaches.extend(
            check_exposure_limits(
                weights,
                limits=exp_lims or None,
                max_gross_exposure=max_gross_exposure,
                max_net_exposure=max_net_exposure,
            )
        )
        breaches.extend(
            check_concentration_limits(
                weights,
                limits=conc_lims or None,
                max_concentration=max_concentration,
            )
        )

    breaches.extend(
        check_loss_limits(
            daily_loss=daily_loss,
            current_drawdown=current_drawdown,
            limits=loss_lims or None,
            max_daily_loss=max_daily_loss,
            max_drawdown=max_drawdown,
        )
    )

    if participation is not None and adv_coverage is not None:
        breaches.extend(
            check_liquidity_limits(
                participation=participation,
                adv_coverage=adv_coverage,
                time_to_liquidate=time_to_liquidate,
                limits=liq_lims or None,
                max_participation=max_participation,
                min_adv_coverage=min_adv_coverage,
            )
        )

    return breaches


__all__ = [
    "LimitSeverity",
    "RiskLimit",
    "LimitBreach",
    "build_position_limits",
    "check_position_limits",
    "check_positions",
    "build_exposure_limits",
    "check_exposure_limits",
    "build_loss_limits",
    "check_loss_limits",
    "build_concentration_limits",
    "check_concentration_limits",
    "build_liquidity_limits",
    "check_liquidity_limits",
    "build_default_limits",
    "check_all_limits",
]
