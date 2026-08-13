"""Build EnsembleDecision from assessments, caps, and hard-limit outcomes."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskState
from iqrp.app.risk.ensemble.config import EnsembleSettings, StateCap
from iqrp.app.risk.ensemble.types import (
    DecisionAction,
    EnsembleDecision,
    RiskAssessment,
    RiskScore,
    utc_now_iso,
)


def state_cap(settings: EnsembleSettings, state: RiskState) -> StateCap:
    caps = settings.state_caps.get(state.value)
    if caps is None:
        # Conservative default if misconfigured
        return StateCap(max_exposure=0.0, recommended_leverage=0.0, position_reduction=1.0)
    return caps


def action_for_state(
    state: RiskState,
    *,
    proposed_exposure: float,
    max_exposure: float,
    fallback_action: DecisionAction | None = None,
    hard_reject: bool = False,
) -> DecisionAction:
    if hard_reject or state == RiskState.TRADING_HALT:
        return DecisionAction.HALT if state == RiskState.TRADING_HALT else DecisionAction.REJECT
    if fallback_action is not None:
        return fallback_action
    if state == RiskState.CAPITAL_PRESERVATION:
        if proposed_exposure > max_exposure + 1e-12:
            return DecisionAction.REJECT
        return DecisionAction.APPROVE_REDUCED if proposed_exposure > 0 else DecisionAction.APPROVE_REDUCED
    if proposed_exposure > max_exposure + 1e-12:
        if state in (RiskState.CAUTION, RiskState.REDUCED_RISK, RiskState.CAPITAL_PRESERVATION):
            return DecisionAction.APPROVE_REDUCED
        return DecisionAction.REJECT
    if state in (RiskState.CAUTION, RiskState.REDUCED_RISK) and proposed_exposure > 0:
        # Still allow but mark reduced when state implies cut
        if state == RiskState.REDUCED_RISK:
            return DecisionAction.APPROVE_REDUCED
    return DecisionAction.APPROVE


def apply_confidence_within_caps(
    *,
    base_leverage: float,
    forecast_confidence: float,
    settings: EnsembleSettings,
    state: RiskState,
) -> float:
    """Scale leverage by forecast confidence only within hard state/leverage caps.

    Forecast confidence / Kelly MUST NOT override hard limits.
    """
    cap = state_cap(settings, state)
    hard_max = min(float(cap.recommended_leverage), float(settings.leverage.max_leverage))
    hard_min = float(settings.leverage.min_leverage)
    if state == RiskState.TRADING_HALT:
        return hard_min

    conf = float(np.clip(forecast_confidence, 0.0, 1.0))
    conf_cap = max(float(settings.leverage.confidence_cap), 1.0)
    # Confidence may only scale within [1/conf_cap, conf_cap] then clip to hard_max
    conf_scalar = 1.0 / conf_cap + conf * (conf_cap - 1.0 / conf_cap)
    conf_scalar = float(np.clip(conf_scalar, 0.0, conf_cap))
    raw = float(base_leverage) * conf_scalar
    # Never exceed state hard cap regardless of confidence
    return float(np.clip(raw, hard_min, hard_max))


def build_decision(
    *,
    settings: EnsembleSettings,
    assessment: RiskAssessment,
    proposed_exposure: float = 0.0,
    forecast_confidence: float = 0.0,
    triggered_limits: list[str] | None = None,
    extra_reasons: list[str] | None = None,
    hard_reject: bool = False,
    hard_reject_reason: str | None = None,
    engine_audit: dict[str, Any] | None = None,
) -> EnsembleDecision:
    state = assessment.risk_state
    cap = state_cap(settings, state)
    max_exp = float(cap.max_exposure)
    # Also respect portfolio gross exposure hard limit
    max_exp = min(max_exp, float(settings.limits.max_gross_exposure))

    fallback_action: DecisionAction | None = None
    if assessment.fallback_applied:
        fallback_action = DecisionAction(settings.missing_metrics_fallback_action)

    limits = list(triggered_limits or [])
    reasons = list(assessment.reasons)
    if extra_reasons:
        reasons.extend(extra_reasons)

    if hard_reject:
        reasons.append(
            hard_reject_reason
            or "REJECTED by underlying RiskIntelligenceEngine hard limits (ensemble soft score cannot override)"
        )
        if "engine_hard_reject" not in limits:
            limits.append("engine_hard_reject")

    if proposed_exposure > max_exp + 1e-12:
        limits.append("max_permitted_exposure")
        reasons.append(
            f"proposed_exposure={proposed_exposure:.4f} exceeds maximum_permitted_exposure={max_exp:.4f} "
            f"for state {state.value}"
        )

    if forecast_confidence > 0 and state in (
        RiskState.TRADING_HALT,
        RiskState.CAPITAL_PRESERVATION,
    ):
        reasons.append(
            f"forecast_confidence={forecast_confidence:.2f} cannot override hard risk state {state.value}"
        )

    decision = action_for_state(
        state,
        proposed_exposure=float(proposed_exposure),
        max_exposure=max_exp,
        fallback_action=fallback_action,
        hard_reject=hard_reject,
    )
    if hard_reject and state != RiskState.TRADING_HALT:
        decision = DecisionAction.REJECT
    if hard_reject and state == RiskState.TRADING_HALT:
        decision = DecisionAction.HALT

    # Leverage: start from state recommendation, allow confidence only within caps
    lev = apply_confidence_within_caps(
        base_leverage=float(cap.recommended_leverage),
        forecast_confidence=float(forecast_confidence),
        settings=settings,
        state=state,
    )
    # Prefer assessment leverage if tighter
    lev = min(lev, float(assessment.recommended_leverage))

    reduction = float(cap.position_reduction)
    if proposed_exposure > max_exp + 1e-12 and proposed_exposure > 1e-12:
        reduction = max(reduction, float(1.0 - max_exp / proposed_exposure))

    ts = utc_now_iso()
    return EnsembleDecision(
        decision=decision,
        risk_state=state,
        risk_score=assessment.dimension_scores,
        risk_confidence=float(assessment.confidence),
        triggered_limits=limits,
        reasons=reasons,
        required_position_reduction=float(np.clip(reduction, 0.0, 1.0)),
        maximum_permitted_exposure=float(max_exp),
        recommended_leverage=float(lev),
        timestamp=ts,
        data_version=settings.data_version,
        model_versions=dict(assessment.risk_model_versions),
        audit={
            "ensemble_version": settings.ensemble_version,
            "fallback_applied": assessment.fallback_applied,
            "missing_critical": list(assessment.missing_critical),
            "proposed_exposure": float(proposed_exposure),
            "forecast_confidence": float(forecast_confidence),
            "state_cap": cap.model_dump(),
            "engine": engine_audit or {},
            "assessment_timestamp": assessment.timestamp,
            "hard_reject": bool(hard_reject),
            "note": "Forecast confidence / Kelly cannot override hard limits",
        },
        proposed_exposure=float(proposed_exposure),
        forecast_confidence=float(forecast_confidence),
    )


def empty_score() -> RiskScore:
    return RiskScore()
