"""Aggregate heterogeneous risk metrics into a unified RiskAssessment.

Does NOT blindly average metrics — preserves dimension identity via scorer + weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskState
from iqrp.app.risk.ensemble.confidence import estimate_confidence
from iqrp.app.risk.ensemble.config import EnsembleSettings
from iqrp.app.risk.ensemble.decision import state_cap
from iqrp.app.risk.ensemble.disagreement import compute_disagreement
from iqrp.app.risk.ensemble.normalizer import normalize_metrics
from iqrp.app.risk.ensemble.scorer import score_dimensions
from iqrp.app.risk.ensemble.state_machine import EnsembleStateMachine
from iqrp.app.risk.ensemble.types import RiskAssessment, utc_now_iso
from iqrp.app.risk.ensemble.weighting import resolve_weights


def missing_critical_keys(metrics: dict[str, Any], settings: EnsembleSettings) -> list[str]:
    missing: list[str] = []
    for key in settings.critical_metric_keys:
        if key not in metrics or metrics[key] is None:
            missing.append(key)
            continue
        val = metrics[key]
        if isinstance(val, dict) and "value" not in val and "score" not in val:
            # allow nested but require a numeric extractable field
            if not any(isinstance(val.get(k), (int, float)) for k in val):
                missing.append(key)
                continue
        try:
            if isinstance(val, dict):
                num = float(val.get("value", val.get("score", float("nan"))))
            else:
                num = float(val)
            if not np.isfinite(num):
                missing.append(key)
        except (TypeError, ValueError):
            missing.append(key)
    return missing


def budget_recommendation(
    *,
    settings: EnsembleSettings,
    overall_score: float,
    state: RiskState,
    confidence: float,
) -> dict[str, Any]:
    """Risk-budget scale from overall score and state; confidence cannot raise above hard caps."""
    cfg = settings.budget
    # Higher overall score → lower budget
    soft = float(np.clip(1.0 - float(overall_score), cfg.min_budget_scale, cfg.max_budget_scale))
    cap = state_cap(settings, state)
    # Hard bind to state exposure cap
    scale = min(soft, float(cap.max_exposure) if cap.max_exposure > 0 else cfg.min_budget_scale)
    # Confidence may only reduce budget further, never expand beyond hard scale
    conf = float(np.clip(confidence, 0.0, 1.0))
    conf_adj = scale * (0.85 + 0.15 * conf)
    conf_adj = min(conf_adj, scale)
    return {
        "target_risk": float(cfg.target_risk),
        "budget_scale": float(conf_adj),
        "soft_scale": float(soft),
        "state_exposure_cap": float(cap.max_exposure),
        "confidence_applied": conf,
        "note": "Confidence cannot expand budget beyond state hard caps",
    }


def aggregate_metrics(
    metrics: dict[str, Any],
    *,
    settings: EnsembleSettings,
    state_machine: EnsembleStateMachine,
    previous_state: RiskState | None = None,
    regime: str = "normal",
    sample_size: int | None = None,
    calibration_stats: dict[str, Any] | None = None,
    weighting_scheme: str | None = None,
    force_fallback_state: bool = False,
) -> RiskAssessment:
    ts = utc_now_iso()
    reasons: list[str] = []
    missing = missing_critical_keys(metrics, settings)
    fallback = bool(missing) or force_fallback_state

    disagreement = compute_disagreement(metrics, settings=settings)
    normalized = normalize_metrics(metrics, settings=settings, timestamp=ts)

    weights = resolve_weights(
        settings,
        scheme=weighting_scheme,  # type: ignore[arg-type]
        disagreement=disagreement,
        calibration_stats=calibration_stats,
        regime=regime,
    )
    # Initial score for dynamic re-weight if needed
    scores = score_dimensions(
        normalized,
        settings=settings,
        weights=weights,
        disagreement=disagreement,
        regime=regime,
    )
    if settings.weighting_scheme in {"dynamic", "risk_budget"} or weighting_scheme in {
        "dynamic",
        "risk_budget",
    }:
        weights = resolve_weights(
            settings,
            scheme=weighting_scheme,  # type: ignore[arg-type]
            dimension_scores=scores.dimension_map(),
            disagreement=disagreement,
            calibration_stats=calibration_stats,
            regime=regime,
        )
        scores = score_dimensions(
            normalized,
            settings=settings,
            weights=weights,
            disagreement=disagreement,
            regime=regime,
        )

    conf = estimate_confidence(
        metrics,
        settings=settings,
        disagreement=float(disagreement.get("overall_disagreement", 0.0) or 0.0),
        sample_size=sample_size,
        missing_critical_count=len(missing),
        as_measure=False,
    )
    assert isinstance(conf, float)

    force_state: RiskState | None = None
    if fallback:
        force_state = RiskState(settings.missing_metrics_fallback_state)
        reasons.append(
            "FALLBACK: missing critical risk metrics "
            + ",".join(missing)
            + f" — applying conservative state {force_state.value}; "
            "do not assume zero risk; auto-approve disabled"
        )

    if disagreement.get("high_disagreement"):
        reasons.append(
            f"High estimator disagreement ({disagreement.get('overall_disagreement'):.3f})"
        )

    state = state_machine.transition(
        scores,
        previous_state=previous_state,
        force_state=force_state,
    )
    if force_state is None:
        reasons.append(
            f"Risk state={state.value} from overall_score={scores.overall:.3f} "
            f"with hysteresis (identity-preserving dimension scores)"
        )

    cap = state_cap(settings, state)
    budget = budget_recommendation(
        settings=settings,
        overall_score=float(scores.overall),
        state=state,
        confidence=float(conf),
    )

    versions = {
        "ensemble": settings.ensemble_version,
        "model": settings.model_version,
        "data": settings.data_version,
    }

    return RiskAssessment(
        timestamp=ts,
        data_version=settings.data_version,
        risk_model_versions=versions,
        input_metrics=dict(metrics),
        normalized_metrics=normalized,
        dimension_scores=scores,
        overall_score=float(scores.overall),
        confidence=float(conf),
        disagreement=disagreement,
        risk_state=state,
        budget_recommendation=budget,
        max_exposure=float(cap.max_exposure),
        recommended_leverage=float(cap.recommended_leverage),
        reasons=reasons,
        missing_critical=missing,
        fallback_applied=fallback,
        audit={
            "weights": weights,
            "regime": regime,
            "state_machine": state_machine.export_state(),
            "weighting_scheme": weighting_scheme or settings.weighting_scheme,
            "sample_size": sample_size,
            "calibration_stats_present": calibration_stats is not None,
        },
    )


class RiskAggregator:
    def __init__(self, settings: EnsembleSettings, state_machine: EnsembleStateMachine) -> None:
        self.settings = settings
        self.state_machine = state_machine

    def aggregate(self, metrics: dict[str, Any], **kwargs: Any) -> RiskAssessment:
        return aggregate_metrics(
            metrics, settings=self.settings, state_machine=self.state_machine, **kwargs
        )
