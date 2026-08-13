"""Diagnostics payloads for the Risk Intelligence Ensemble."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.ensemble.types import EnsembleDecision, RiskAssessment, RiskScore


def dimension_breakdown(scores: RiskScore) -> dict[str, Any]:
    dims = scores.dimension_map()
    weights = dict(scores.weights_applied)
    contributors = dict(scores.contributors)
    ranked = sorted(dims.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "dimensions": dims,
        "weights": weights,
        "contributors": contributors,
        "ranked_by_risk": [{"dimension": k, "score": v} for k, v in ranked],
        "overall": float(scores.overall),
        "top_driver": ranked[0][0] if ranked else None,
        "metadata": dict(scores.metadata),
    }


def assessment_diagnostics(assessment: RiskAssessment) -> dict[str, Any]:
    return {
        "timestamp": assessment.timestamp,
        "risk_state": assessment.risk_state.value,
        "overall_score": assessment.overall_score,
        "confidence": assessment.confidence,
        "fallback_applied": assessment.fallback_applied,
        "missing_critical": list(assessment.missing_critical),
        "dimension_breakdown": dimension_breakdown(assessment.dimension_scores),
        "disagreement_summary": {
            "overall": assessment.disagreement.get("overall_disagreement"),
            "uncertainty": assessment.disagreement.get("overall_uncertainty"),
            "high": assessment.disagreement.get("high_disagreement"),
            "n_pairs": assessment.disagreement.get("n_pairs_available"),
        },
        "normalized_count": len(assessment.normalized_metrics),
        "budget": dict(assessment.budget_recommendation),
        "max_exposure": assessment.max_exposure,
        "recommended_leverage": assessment.recommended_leverage,
        "reasons": list(assessment.reasons),
    }


def decision_diagnostics(decision: EnsembleDecision) -> dict[str, Any]:
    return {
        "decision": decision.decision.value,
        "risk_state": decision.risk_state.value,
        "risk_confidence": decision.risk_confidence,
        "triggered_limits": list(decision.triggered_limits),
        "required_position_reduction": decision.required_position_reduction,
        "maximum_permitted_exposure": decision.maximum_permitted_exposure,
        "recommended_leverage": decision.recommended_leverage,
        "proposed_exposure": decision.proposed_exposure,
        "forecast_confidence": decision.forecast_confidence,
        "hard_limits_note": "Forecast confidence cannot override hard limits",
        "reasons": list(decision.reasons),
        "score_overall": decision.risk_score.overall,
        "audit_keys": sorted(decision.audit.keys()),
    }


def health_check(
    *,
    assessment: RiskAssessment | None = None,
    decision: EnsembleDecision | None = None,
    state_machine_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    if assessment is not None and assessment.fallback_applied:
        issues.append("critical_metrics_missing_fallback_active")
    if assessment is not None and assessment.disagreement.get("high_disagreement"):
        issues.append("high_estimator_disagreement")
    if decision is not None and decision.decision.value in {"REJECT", "HALT"}:
        issues.append(f"decision_{decision.decision.value.lower()}")
    status = "ok" if not issues else "degraded"
    return {
        "status": status,
        "issues": issues,
        "assessment": assessment_diagnostics(assessment) if assessment else None,
        "decision": decision_diagnostics(decision) if decision else None,
        "state_machine": state_machine_state,
    }


class EnsembleDiagnostics:
    def breakdown(self, scores: RiskScore) -> dict[str, Any]:
        return dimension_breakdown(scores)

    def for_assessment(self, assessment: RiskAssessment) -> dict[str, Any]:
        return assessment_diagnostics(assessment)

    def for_decision(self, decision: EnsembleDecision) -> dict[str, Any]:
        return decision_diagnostics(decision)

    def health(self, **kwargs: Any) -> dict[str, Any]:
        return health_check(**kwargs)
