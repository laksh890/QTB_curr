"""Visualization payloads for ensemble scores and state (no matplotlib required)."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.base import RiskState
from iqrp.app.risk.ensemble.types import EnsembleDecision, RiskAssessment, RiskScore

_STATE_ORDER = [
    RiskState.NORMAL.value,
    RiskState.CAUTION.value,
    RiskState.REDUCED_RISK.value,
    RiskState.CAPITAL_PRESERVATION.value,
    RiskState.TRADING_HALT.value,
]


def score_radar_payload(scores: RiskScore) -> dict[str, Any]:
    dims = scores.dimension_map()
    return {
        "type": "radar",
        "title": "Risk Dimension Scores",
        "axes": list(dims.keys()),
        "values": [float(dims[k]) for k in dims],
        "overall": float(scores.overall),
        "range": [0.0, 1.0],
        "weights": dict(scores.weights_applied),
    }


def score_bars_payload(scores: RiskScore) -> dict[str, Any]:
    dims = scores.dimension_map()
    ranked = sorted(dims.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "type": "bar",
        "title": "Dimension Risk (1 = max)",
        "categories": [k for k, _ in ranked],
        "values": [float(v) for _, v in ranked],
        "contributors": [float(scores.contributors.get(k, 0.0)) for k, _ in ranked],
        "range": [0.0, 1.0],
    }


def state_gauge_payload(state: RiskState | str, *, overall_score: float | None = None) -> dict[str, Any]:
    value = state.value if isinstance(state, RiskState) else str(state)
    idx = _STATE_ORDER.index(value) if value in _STATE_ORDER else 0
    return {
        "type": "gauge",
        "title": "Ensemble Risk State",
        "state": value,
        "state_index": idx,
        "states": list(_STATE_ORDER),
        "overall_score": overall_score,
    }


def disagreement_payload(disagreement: dict[str, Any]) -> dict[str, Any]:
    pairs = disagreement.get("pairs") or []
    return {
        "type": "grouped_bar",
        "title": "Estimator Disagreement",
        "overall_disagreement": disagreement.get("overall_disagreement"),
        "overall_uncertainty": disagreement.get("overall_uncertainty"),
        "high_disagreement": disagreement.get("high_disagreement"),
        "categories": [p.get("name") or str(p.get("pair")) for p in pairs if p.get("available")],
        "disagreement": [float(p["disagreement"]) for p in pairs if p.get("available")],
        "uncertainty": [float(p["uncertainty"]) for p in pairs if p.get("available")],
    }


def assessment_viz(assessment: RiskAssessment) -> dict[str, Any]:
    return {
        "score_radar": score_radar_payload(assessment.dimension_scores),
        "score_bars": score_bars_payload(assessment.dimension_scores),
        "state_gauge": state_gauge_payload(assessment.risk_state, overall_score=assessment.overall_score),
        "disagreement": disagreement_payload(assessment.disagreement),
        "meta": {
            "timestamp": assessment.timestamp,
            "confidence": assessment.confidence,
            "fallback_applied": assessment.fallback_applied,
            "max_exposure": assessment.max_exposure,
            "recommended_leverage": assessment.recommended_leverage,
        },
    }


def decision_viz(decision: EnsembleDecision) -> dict[str, Any]:
    return {
        "state_gauge": state_gauge_payload(decision.risk_state, overall_score=decision.risk_score.overall),
        "score_bars": score_bars_payload(decision.risk_score),
        "decision_card": {
            "type": "card",
            "decision": decision.decision.value,
            "risk_state": decision.risk_state.value,
            "maximum_permitted_exposure": decision.maximum_permitted_exposure,
            "recommended_leverage": decision.recommended_leverage,
            "required_position_reduction": decision.required_position_reduction,
            "triggered_limits": list(decision.triggered_limits),
            "reasons": list(decision.reasons),
        },
    }


class EnsembleVisualization:
    def scores(self, scores: RiskScore) -> dict[str, Any]:
        return {"radar": score_radar_payload(scores), "bars": score_bars_payload(scores)}

    def state(self, state: RiskState | str, *, overall_score: float | None = None) -> dict[str, Any]:
        return state_gauge_payload(state, overall_score=overall_score)

    def assessment(self, assessment: RiskAssessment) -> dict[str, Any]:
        return assessment_viz(assessment)

    def decision(self, decision: EnsembleDecision) -> dict[str, Any]:
        return decision_viz(decision)
