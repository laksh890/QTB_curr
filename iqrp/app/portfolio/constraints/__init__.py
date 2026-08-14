"""Portfolio constraint checkers.

``check_all_constraints`` aggregates all constraint modules. Hard constraints are
reported and **never** auto-relaxed. Soft constraints are tagged with soft severity
and may be ignored by callers; this function does not modify weights.
"""

from __future__ import annotations

from typing import Any

from iqrp.app.portfolio.constraints._types import (
    ConstraintSeverity,
    ConstraintViolation,
    coerce_severity,
    filter_by_severity,
)
from iqrp.app.portfolio.constraints.beta import check_beta_constraints, portfolio_beta
from iqrp.app.portfolio.constraints.concentration import (
    check_concentration_constraints,
    concentration_metrics,
)
from iqrp.app.portfolio.constraints.currency import (
    check_currency_constraints,
    currency_exposures,
)
from iqrp.app.portfolio.constraints.exposure import (
    check_exposure_constraints,
    exposure_metrics,
)
from iqrp.app.portfolio.constraints.factor import (
    check_factor_constraints,
    portfolio_factor_exposures,
)
from iqrp.app.portfolio.constraints.leverage import (
    check_leverage_constraints,
    leverage_metrics,
)
from iqrp.app.portfolio.constraints.liquidity import check_liquidity_constraints
from iqrp.app.portfolio.constraints.position import check_position_constraints
from iqrp.app.portfolio.constraints.risk import check_risk_constraints
from iqrp.app.portfolio.constraints.sector import check_sector_constraints, sector_exposures
from iqrp.app.portfolio.constraints.turnover import check_turnover_constraints, turnover


def check_all_constraints(weights: Any, **kwargs: Any) -> list[ConstraintViolation]:
    """Run enabled constraint checks and return all violations.

    Only constraints with explicit limit kwargs (or required side inputs) are
    evaluated. Soft vs hard is controlled by ``severity``, ``hard``, or
    ``soft_constraints`` (iterable of constraint name prefixes).

    **Hard constraints are never auto-relaxed** — this function only reports.
    """
    soft_names = {str(x).lower() for x in (kwargs.get("soft_constraints") or [])}
    default_sev = coerce_severity(kwargs.get("severity"), hard=kwargs.get("hard"))

    def _sev(name: str) -> ConstraintSeverity:
        if name.lower() in soft_names or any(name.lower().startswith(s) for s in soft_names):
            return ConstraintSeverity.SOFT
        return default_sev

    violations: list[ConstraintViolation] = []

    # Exposure
    if any(k in kwargs for k in ("max_gross", "max_net", "min_net", "max_long", "max_short")):
        violations.extend(
            check_exposure_constraints(
                weights,
                max_gross=kwargs.get("max_gross"),
                max_net=kwargs.get("max_net"),
                min_net=kwargs.get("min_net"),
                max_long=kwargs.get("max_long"),
                max_short=kwargs.get("max_short"),
                severity=_sev("max_gross_exposure"),
            )
        )

    # Leverage
    if any(k in kwargs for k in ("max_leverage", "min_leverage")):
        violations.extend(
            check_leverage_constraints(
                weights,
                max_leverage=kwargs.get("max_leverage"),
                min_leverage=kwargs.get("min_leverage"),
                severity=_sev("max_leverage"),
            )
        )

    # Concentration (max_weight / HHI / effective N)
    if any(k in kwargs for k in ("max_weight", "max_hhi", "min_effective_n", "max_herfindahl")):
        violations.extend(
            check_concentration_constraints(
                weights,
                max_weight=kwargs.get("max_weight"),
                max_hhi=kwargs.get("max_hhi", kwargs.get("max_herfindahl")),
                min_effective_n=kwargs.get("min_effective_n"),
                severity=_sev("max_weight"),
            )
        )

    # Position box / long-only
    if any(k in kwargs for k in ("max_position", "min_position", "min_weight", "long_only")):
        violations.extend(
            check_position_constraints(
                weights,
                max_position=kwargs.get("max_position"),
                min_position=kwargs.get("min_position"),
                max_weight=kwargs.get("max_weight") if "max_position" not in kwargs else None,
                min_weight=kwargs.get("min_weight"),
                long_only=bool(kwargs.get("long_only", False)),
                severity=_sev("max_position"),
            )
        )

    # Liquidity
    if kwargs.get("adv") is not None and any(
        k in kwargs
        for k in (
            "max_participation",
            "max_participation_rate",
            "max_ttl",
            "min_adv_coverage",
        )
    ):
        violations.extend(
            check_liquidity_constraints(
                weights,
                adv=kwargs.get("adv"),
                spreads=kwargs.get("spreads"),
                prices=kwargs.get("prices"),
                vols=kwargs.get("vols"),
                capital=float(kwargs.get("capital", 1.0)),
                max_participation=kwargs.get("max_participation"),
                max_participation_rate=kwargs.get("max_participation_rate"),
                max_ttl=kwargs.get("max_ttl"),
                min_adv_coverage=kwargs.get("min_adv_coverage"),
                impact_coeff=float(kwargs.get("impact_coeff", 0.1)),
                severity=_sev("max_participation"),
            )
        )

    # Turnover
    if any(k in kwargs for k in ("max_turnover", "min_trade")) and (
        kwargs.get("weights_old") is not None or kwargs.get("current_weights") is not None
    ):
        violations.extend(
            check_turnover_constraints(
                weights,
                weights_old=kwargs.get("weights_old"),
                current_weights=kwargs.get("current_weights"),
                max_turnover=kwargs.get("max_turnover"),
                min_trade=kwargs.get("min_trade"),
                severity=_sev("max_turnover"),
            )
        )

    # Sector
    if kwargs.get("sector_map") is not None and any(
        k in kwargs for k in ("max_sector_weight", "min_sector_weight")
    ):
        violations.extend(
            check_sector_constraints(
                weights,
                sector_map=kwargs.get("sector_map"),
                names=kwargs.get("names"),
                max_sector_weight=kwargs.get("max_sector_weight"),
                min_sector_weight=kwargs.get("min_sector_weight"),
                severity=_sev("max_sector_weight"),
            )
        )

    # Factor
    if kwargs.get("factor_loadings") is not None and any(
        k in kwargs
        for k in (
            "max_factor_exposure",
            "min_factor_exposure",
            "factor_neutral",
        )
    ):
        violations.extend(
            check_factor_constraints(
                weights,
                factor_loadings=kwargs.get("factor_loadings"),
                factor_names=kwargs.get("factor_names"),
                max_factor_exposure=kwargs.get("max_factor_exposure"),
                min_factor_exposure=kwargs.get("min_factor_exposure"),
                factor_neutral=kwargs.get("factor_neutral", False),
                neutrality_tol=float(kwargs.get("neutrality_tol", 1e-6)),
                severity=_sev("max_factor_exposure"),
            )
        )

    # Currency
    if kwargs.get("currencies") is not None and any(
        k in kwargs for k in ("max_currency_exposure", "min_currency_exposure")
    ):
        violations.extend(
            check_currency_constraints(
                weights,
                currencies=kwargs.get("currencies"),
                max_currency_exposure=kwargs.get("max_currency_exposure"),
                min_currency_exposure=kwargs.get("min_currency_exposure"),
                severity=_sev("max_currency_exposure"),
            )
        )

    # Beta
    if any(k in kwargs for k in ("max_beta", "min_beta", "target_beta")) and (
        kwargs.get("betas") is not None or kwargs.get("portfolio_beta_value") is not None
    ):
        violations.extend(
            check_beta_constraints(
                weights,
                betas=kwargs.get("betas"),
                portfolio_beta_value=kwargs.get("portfolio_beta_value"),
                max_beta=kwargs.get("max_beta"),
                min_beta=kwargs.get("min_beta"),
                target_beta=kwargs.get("target_beta"),
                beta_tol=float(kwargs.get("beta_tol", 0.05)),
                severity=_sev("max_beta"),
            )
        )

    # Risk metrics (precomputed)
    if any(
        k in kwargs
        for k in (
            "max_var",
            "max_cvar",
            "max_expected_shortfall",
            "max_drawdown",
            "max_risk_contribution",
        )
    ):
        violations.extend(
            check_risk_constraints(
                weights,
                var=kwargs.get("var"),
                cvar=kwargs.get("cvar"),
                expected_shortfall=kwargs.get("expected_shortfall"),
                drawdown=kwargs.get("drawdown"),
                risk_contribution=kwargs.get("risk_contribution"),
                max_var=kwargs.get("max_var"),
                max_cvar=kwargs.get("max_cvar"),
                max_expected_shortfall=kwargs.get("max_expected_shortfall"),
                max_drawdown=kwargs.get("max_drawdown"),
                max_risk_contribution=kwargs.get("max_risk_contribution"),
                risk_metrics=kwargs.get("risk_metrics"),
                severity=_sev("max_var"),
            )
        )

    include_soft = bool(kwargs.get("include_soft", True))
    include_hard = bool(kwargs.get("include_hard", True))
    return filter_by_severity(violations, include_soft=include_soft, include_hard=include_hard)


__all__ = [
    "ConstraintSeverity",
    "ConstraintViolation",
    "check_all_constraints",
    "check_beta_constraints",
    "check_concentration_constraints",
    "check_currency_constraints",
    "check_exposure_constraints",
    "check_factor_constraints",
    "check_leverage_constraints",
    "check_liquidity_constraints",
    "check_position_constraints",
    "check_risk_constraints",
    "check_sector_constraints",
    "check_turnover_constraints",
    "concentration_metrics",
    "currency_exposures",
    "exposure_metrics",
    "filter_by_severity",
    "leverage_metrics",
    "portfolio_beta",
    "portfolio_factor_exposures",
    "sector_exposures",
    "turnover",
]
