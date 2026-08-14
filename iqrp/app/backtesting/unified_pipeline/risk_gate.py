"""Risk and position-sizing handoffs to existing RiskIntelligenceEngine."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from iqrp.app.backtesting.unified_pipeline.types import (
    AlphaCandidate,
    RiskHandoffResult,
    SizingResult,
    StageOutcome,
)
from iqrp.app.risk import RiskIntelligenceEngine, RiskSettings


def _decision_id() -> str:
    return f"risk_{uuid.uuid4().hex[:16]}"


def evaluate_candidate_risk(
    candidate: AlphaCandidate,
    *,
    risk_engine: RiskIntelligenceEngine,
    current_weights: dict[str, float],
    returns: Any,
    instrument_order: list[str] | None = None,
    forecast_confidence: float | None = None,
) -> RiskHandoffResult:
    """Gate a candidate with existing validate_position; map to APPROVED/REDUCED/REJECTED."""
    names = list(instrument_order or sorted(set(current_weights) | {candidate.instrument}))
    if candidate.instrument not in names:
        names.append(candidate.instrument)
    weights = np.array([float(current_weights.get(n, 0.0)) for n in names], dtype=np.float64)
    idx = names.index(candidate.instrument)
    requested = float(
        candidate.requested_weight
        if candidate.requested_weight is not None
        else candidate.direction * 0.05
    )
    conf = float(
        forecast_confidence
        if forecast_confidence is not None
        else (candidate.confidence if candidate.confidence is not None else 0.0)
    )
    decision = risk_engine.validate_position(
        proposed_weight=requested,
        weights=weights,
        returns=returns,
        forecast_confidence=conf,
        asset_index=idx,
    )
    breaches = [b.to_dict() for b in decision.breaches]
    codes = [b.get("limit_name", "limit") for b in breaches]
    rid = _decision_id()
    if not decision.approved:
        return RiskHandoffResult(
            risk_decision_id=rid,
            outcome=StageOutcome.RISK_REJECTED,
            requested_exposure=requested,
            approved_exposure=float(current_weights.get(candidate.instrument, 0.0)),
            reason=decision.reason,
            reason_codes=codes or ["RISK_REJECTED"],
            limits_triggered=breaches,
            recommended_size=decision.recommended_size,
            audit=dict(decision.audit),
        )

    # Size toward recommended when smaller than request → REDUCED
    approved = requested
    outcome = StageOutcome.RISK_APPROVED
    if decision.recommended_size is not None and abs(requested) > 1e-15:
        rec = float(decision.recommended_size)
        max_pos = float(risk_engine.settings.limits.max_position)
        cap = min(rec, max_pos)
        if abs(requested) > cap + 1e-12:
            approved = float(np.sign(requested)) * cap
            outcome = StageOutcome.RISK_REDUCED
            codes = list(codes) + ["SIZE_REDUCED_TO_RECOMMENDED"]
    if "APPROVED_WITH_WARNINGS" in decision.reason and outcome == StageOutcome.RISK_APPROVED:
        # keep APPROVED but record warnings; optional soft reduction already applied above
        pass

    return RiskHandoffResult(
        risk_decision_id=rid,
        outcome=outcome,
        requested_exposure=requested,
        approved_exposure=float(approved),
        reason=decision.reason,
        reason_codes=codes,
        limits_triggered=breaches,
        recommended_size=decision.recommended_size,
        audit=dict(decision.audit),
    )


def size_approved_exposure(
    *,
    risk_engine: RiskIntelligenceEngine,
    approved_exposure: float,
    returns: Any,
    equity: float = 1.0,
    confidence: float = 1.0,
    method: str | None = None,
) -> SizingResult:
    """Apply configured RiskIntelligenceEngine.position_size; do not pick a 'best' method."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    vol = float(np.std(r[-60:]) if r.size else 0.02) or 0.02
    sizing = risk_engine.position_size(
        realized_vol=vol,
        confidence=confidence,
        equity=equity,
        method=method,
    )
    raw = float(sizing["size"])
    requested = float(approved_exposure)
    # Combine: final signed size is sign(request) * min(|request|, raw) when request nonzero
    if abs(requested) < 1e-15:
        final = 0.0
    else:
        final = float(np.sign(requested)) * min(abs(requested), abs(raw) if raw > 0 else abs(requested))
    return SizingResult(
        requested_size=requested,
        risk_adjusted_size=float(np.sign(requested)) * abs(raw) if requested != 0 else 0.0,
        final_size=final,
        sizing_method=str(sizing.get("method", method or risk_engine.settings.sizing.method)),
        sizing_configuration={
            "target_volatility": risk_engine.settings.sizing.target_volatility,
            "max_leverage": risk_engine.settings.sizing.max_leverage,
            "engine_output": {k: sizing[k] for k in ("method", "size", "note") if k in sizing},
        },
    )


def default_risk_engine(*, max_position: float | None = None, max_leverage: float | None = None) -> RiskIntelligenceEngine:
    settings = RiskSettings.default()
    updates: dict[str, Any] = {}
    if max_position is not None:
        updates["limits"] = settings.limits.model_copy(update={"max_position": float(max_position)})
    if max_leverage is not None:
        lim = updates.get("limits", settings.limits)
        updates["limits"] = lim.model_copy(update={"max_leverage": float(max_leverage)})
        updates["sizing"] = settings.sizing.model_copy(update={"max_leverage": float(max_leverage)})
    if updates:
        settings = settings.model_copy(update=updates)
    return RiskIntelligenceEngine(settings=settings)


__all__ = [
    "default_risk_engine",
    "evaluate_candidate_risk",
    "size_approved_exposure",
]
