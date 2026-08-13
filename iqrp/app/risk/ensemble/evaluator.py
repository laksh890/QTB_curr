"""Evaluation helpers for ensemble quality, stability, and gate outcomes."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskState
from iqrp.app.risk.ensemble.calibration import run_calibration
from iqrp.app.risk.ensemble.config import EnsembleSettings
from iqrp.app.risk.ensemble.types import DecisionAction, EnsembleDecision, RiskAssessment


def evaluate_decision_consistency(decisions: list[EnsembleDecision]) -> dict[str, Any]:
    if not decisions:
        return {"n": 0, "approve_rate": None, "halt_rate": None, "reject_rate": None}
    n = len(decisions)
    counts = {a.value: 0 for a in DecisionAction}
    for d in decisions:
        counts[d.decision.value] = counts.get(d.decision.value, 0) + 1
    states = [d.risk_state.value for d in decisions]
    flips = sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])
    return {
        "n": n,
        "counts": counts,
        "approve_rate": counts.get(DecisionAction.APPROVE.value, 0) / n,
        "approve_reduced_rate": counts.get(DecisionAction.APPROVE_REDUCED.value, 0) / n,
        "reject_rate": counts.get(DecisionAction.REJECT.value, 0) / n,
        "halt_rate": counts.get(DecisionAction.HALT.value, 0) / n,
        "state_flip_count": flips,
        "state_flip_rate": flips / max(n - 1, 1),
    }


def evaluate_score_stability(scores: list[float]) -> dict[str, Any]:
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "max_abs_delta": None}
    deltas = np.diff(arr) if arr.size > 1 else np.asarray([0.0])
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "max_abs_delta": float(np.max(np.abs(deltas))) if deltas.size else 0.0,
    }


def evaluate_fallback_rate(assessments: list[RiskAssessment]) -> dict[str, Any]:
    n = len(assessments)
    if n == 0:
        return {"n": 0, "fallback_rate": None}
    fb = sum(1 for a in assessments if a.fallback_applied)
    return {"n": n, "fallback_count": fb, "fallback_rate": fb / n}


def evaluate_hard_limit_override_attempts(decisions: list[EnsembleDecision]) -> dict[str, Any]:
    """Count cases where forecast confidence was present under halt/preservation (must not approve)."""
    flagged = []
    for d in decisions:
        if d.forecast_confidence > 0 and d.risk_state in (
            RiskState.TRADING_HALT,
            RiskState.CAPITAL_PRESERVATION,
        ):
            if d.decision in (DecisionAction.APPROVE,):
                flagged.append(d.timestamp)
    return {
        "override_violations": len(flagged),
        "timestamps": flagged,
        "rule": "forecast_confidence_must_not_override_hard_limits",
        "ok": len(flagged) == 0,
    }


def evaluate_ensemble(
    *,
    settings: EnsembleSettings,
    assessments: list[RiskAssessment] | None = None,
    decisions: list[EnsembleDecision] | None = None,
    calibration_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assessments = assessments or []
    decisions = decisions or []
    out: dict[str, Any] = {
        "decision_consistency": evaluate_decision_consistency(decisions),
        "score_stability": evaluate_score_stability([a.overall_score for a in assessments]),
        "fallback": evaluate_fallback_rate(assessments),
        "hard_limit_guard": evaluate_hard_limit_override_attempts(decisions),
    }
    if calibration_kwargs:
        out["calibration"] = run_calibration(settings=settings, **calibration_kwargs)
    return out


class EnsembleEvaluator:
    def __init__(self, settings: EnsembleSettings) -> None:
        self.settings = settings

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        return evaluate_ensemble(settings=self.settings, **kwargs)
