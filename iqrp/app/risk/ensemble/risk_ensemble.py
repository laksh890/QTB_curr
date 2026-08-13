"""Risk Intelligence Ensemble — unified multi-dimension risk gate.

Architectural rules:
1. Do not blindly average metrics; preserve dimension identity.
2. Missing critical risk info → conservative fallback (never assume zero / auto-approve).
3. Forecast confidence / Kelly MUST NOT override hard limits.
4. State transitions are deterministic with hysteresis.
5. Single noisy metric must not trigger TRADING_HALT unless hard_halt_on_single.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskState, as_returns, as_weights
from iqrp.app.risk.ensemble.aggregator import RiskAggregator, aggregate_metrics, missing_critical_keys
from iqrp.app.risk.ensemble.calibration import CalibrationEngine, run_calibration
from iqrp.app.risk.ensemble.confidence import ConfidenceEstimator, estimate_confidence
from iqrp.app.risk.ensemble.config import EnsembleSettings
from iqrp.app.risk.ensemble.decision import build_decision, state_cap
from iqrp.app.risk.ensemble.diagnostics import EnsembleDiagnostics
from iqrp.app.risk.ensemble.disagreement import DisagreementAnalyzer, compute_disagreement
from iqrp.app.risk.ensemble.evaluator import EnsembleEvaluator
from iqrp.app.risk.ensemble.normalizer import MetricNormalizer, normalize_metrics
from iqrp.app.risk.ensemble.scorer import RiskScorer, score_dimensions
from iqrp.app.risk.ensemble.serializer import EnsembleSerializer
from iqrp.app.risk.ensemble.state_machine import EnsembleStateMachine
from iqrp.app.risk.ensemble.types import (
    DecisionAction,
    EnsembleDecision,
    RiskAssessment,
    RiskScore,
)
from iqrp.app.risk.ensemble.visualization import EnsembleVisualization
from iqrp.app.risk.ensemble.weighting import WeightResolver
from iqrp.app.risk.leverage.dynamic_leverage import recommended_leverage as eng_recommended_leverage


class RiskIntelligenceEnsemble:
    """Multi-dimension risk ensemble coordinating scoring, state, and decisions."""

    def __init__(
        self,
        settings: EnsembleSettings | None = None,
        risk_engine: Any | None = None,
    ) -> None:
        self.settings = settings or EnsembleSettings.default()
        self.risk_engine = risk_engine
        self._state_machine = EnsembleStateMachine(self.settings)
        self._normalizer = MetricNormalizer(self.settings)
        self._scorer = RiskScorer(self.settings)
        self._weights = WeightResolver(self.settings)
        self._confidence = ConfidenceEstimator(self.settings)
        self._disagreement = DisagreementAnalyzer(self.settings)
        self._calibration = CalibrationEngine(self.settings)
        self._aggregator = RiskAggregator(self.settings, self._state_machine)
        self._diagnostics = EnsembleDiagnostics()
        self._evaluator = EnsembleEvaluator(self.settings)
        self._visualization = EnsembleVisualization()
        self._serializer = EnsembleSerializer()
        self._last_assessment: RiskAssessment | None = None
        self._last_decision: EnsembleDecision | None = None
        self._audit_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ API
    def aggregate(self, metrics: dict[str, Any], **kwargs: Any) -> RiskAssessment:
        assessment = aggregate_metrics(
            metrics,
            settings=self.settings,
            state_machine=self._state_machine,
            **kwargs,
        )
        self._last_assessment = assessment
        self._audit_log.append(
            {
                "event": "aggregate",
                "timestamp": assessment.timestamp,
                "risk_state": assessment.risk_state.value,
                "overall_score": assessment.overall_score,
                "fallback_applied": assessment.fallback_applied,
                "missing_critical": list(assessment.missing_critical),
            }
        )
        return assessment

    def score(self, metrics: dict[str, Any] | RiskAssessment) -> RiskScore:
        if isinstance(metrics, RiskAssessment):
            return metrics.dimension_scores
        normalized = normalize_metrics(metrics, settings=self.settings)
        disagreement = compute_disagreement(metrics, settings=self.settings)
        weights = self._weights.resolve(disagreement=disagreement)
        return score_dimensions(
            normalized,
            settings=self.settings,
            weights=weights,
            disagreement=disagreement,
        )

    def confidence(
        self,
        metrics: dict[str, Any],
        *,
        disagreement: float | None = None,
        sample_size: int | None = None,
        as_measure: bool = False,
        **kwargs: Any,
    ) -> float | Any:
        missing = missing_critical_keys(metrics, self.settings)
        if disagreement is None:
            disc = compute_disagreement(metrics, settings=self.settings)
            disagreement = float(disc.get("overall_disagreement", 0.0) or 0.0)
        return estimate_confidence(
            metrics,
            settings=self.settings,
            disagreement=disagreement,
            sample_size=sample_size,
            missing_critical_count=len(missing),
            as_measure=as_measure,
            **kwargs,
        )

    def disagreement(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return compute_disagreement(metrics, settings=self.settings)

    def risk_state(
        self,
        scores: RiskScore | dict[str, Any],
        *,
        previous_state: RiskState | None = None,
    ) -> RiskState:
        return self._state_machine.transition(scores, previous_state=previous_state)

    def decision(
        self,
        *,
        assessment: RiskAssessment | None = None,
        metrics: dict[str, Any] | None = None,
        proposed_exposure: float = 0.0,
        forecast_confidence: float = 0.0,
        triggered_limits: list[str] | None = None,
        extra_reasons: list[str] | None = None,
        hard_reject: bool = False,
        hard_reject_reason: str | None = None,
        engine_audit: dict[str, Any] | None = None,
        **aggregate_kwargs: Any,
    ) -> EnsembleDecision:
        if assessment is None:
            if metrics is None:
                raise ValueError("decision() requires assessment= or metrics=")
            assessment = self.aggregate(metrics, **aggregate_kwargs)

        # Missing critical → never APPROVE
        if assessment.fallback_applied and assessment.risk_state == RiskState(
            self.settings.missing_metrics_fallback_state
        ):
            # Ensure fallback action is not APPROVE
            pass

        decision = build_decision(
            settings=self.settings,
            assessment=assessment,
            proposed_exposure=float(proposed_exposure),
            forecast_confidence=float(forecast_confidence),
            triggered_limits=triggered_limits,
            extra_reasons=extra_reasons,
            hard_reject=hard_reject,
            hard_reject_reason=hard_reject_reason,
            engine_audit=engine_audit,
        )

        # Absolute guard: missing critical metrics cannot yield APPROVE
        if assessment.fallback_applied and decision.decision == DecisionAction.APPROVE:
            decision = EnsembleDecision(
                decision=DecisionAction(self.settings.missing_metrics_fallback_action),
                risk_state=assessment.risk_state,
                risk_score=decision.risk_score,
                risk_confidence=decision.risk_confidence,
                triggered_limits=list(decision.triggered_limits) + ["missing_critical_metrics"],
                reasons=list(decision.reasons)
                + ["GUARD: missing critical metrics — auto-approve disabled"],
                required_position_reduction=decision.required_position_reduction,
                maximum_permitted_exposure=decision.maximum_permitted_exposure,
                recommended_leverage=decision.recommended_leverage,
                timestamp=decision.timestamp,
                data_version=decision.data_version,
                model_versions=dict(decision.model_versions),
                audit={**decision.audit, "approve_blocked_missing_critical": True},
                proposed_exposure=decision.proposed_exposure,
                forecast_confidence=decision.forecast_confidence,
            )

        self._last_decision = decision
        self._audit_log.append(decision.to_dict())
        return decision

    def validate_position(
        self,
        *,
        proposed_weight: float,
        weights: Any,
        returns: Any,
        metrics: dict[str, Any] | None = None,
        forecast_confidence: float = 0.0,
        realized_vol: float | None = None,
        participation: float | None = None,
        adv_coverage: float | None = None,
        asset_index: int = 0,
        **kwargs: Any,
    ) -> EnsembleDecision:
        """Pre-trade gate.

        Flow:
        1. Missing critical keys → conservative REJECT/HALT fallback logged
        2. aggregate / score / state / decision
        3. If risk_engine present, call eng.validate_position; engine hard rejects win
        4. Apply exposure/leverage caps from state
        """
        w = as_weights(weights).copy()
        if 0 <= asset_index < w.size:
            w[asset_index] = float(proposed_weight)
        else:
            w = np.append(w, float(proposed_weight))
        proposed_exposure = float(np.sum(np.abs(w)))

        metric_bag: dict[str, Any] = dict(metrics or {})
        # Enrich from returns / risk engine when metrics incomplete (PIT, no future info).
        r = as_returns(returns)
        if "drawdown" not in metric_bag and r.size:
            from iqrp.app.risk.tail.drawdown import drawdown_series

            dd = drawdown_series(r)
            metric_bag["drawdown"] = float(dd[-1]) if dd.size else None
        if "volatility" not in metric_bag:
            if realized_vol is not None:
                metric_bag["volatility"] = float(realized_vol)
            elif r.size:
                from iqrp.app.risk.market.volatility import realized_volatility

                metric_bag["volatility"] = float(realized_volatility(r).value)
        if r.size and self.risk_engine is not None:
            if "var" not in metric_bag and hasattr(self.risk_engine, "var"):
                metric_bag["var"] = float(self.risk_engine.var(r).value)
            if "cvar" not in metric_bag and hasattr(self.risk_engine, "cvar"):
                metric_bag["cvar"] = float(self.risk_engine.cvar(r).value)
            if "expected_shortfall" not in metric_bag and hasattr(self.risk_engine, "expected_shortfall"):
                metric_bag["expected_shortfall"] = float(self.risk_engine.expected_shortfall(r).value)
        elif r.size:
            from iqrp.app.risk.tail.cvar import historical_cvar
            from iqrp.app.risk.tail.var import historical_var

            if "var" not in metric_bag:
                metric_bag["var"] = float(historical_var(r).value)
            if "cvar" not in metric_bag:
                metric_bag["cvar"] = float(historical_cvar(r).value)

        missing = missing_critical_keys(metric_bag, self.settings)
        assessment = self.aggregate(metric_bag, **kwargs)

        hard_reject = False
        hard_reason: str | None = None
        engine_audit: dict[str, Any] = {}
        engine_state: RiskState | None = None

        if self.risk_engine is not None and hasattr(self.risk_engine, "validate_position"):
            eng_decision = self.risk_engine.validate_position(
                proposed_weight=float(proposed_weight),
                weights=weights,
                returns=returns,
                realized_vol=realized_vol,
                participation=participation,
                adv_coverage=adv_coverage,
                forecast_confidence=float(forecast_confidence),
                asset_index=int(asset_index),
            )
            engine_audit = eng_decision.to_dict() if hasattr(eng_decision, "to_dict") else {"raw": str(eng_decision)}
            engine_state = getattr(eng_decision, "risk_state", None)
            approved = bool(getattr(eng_decision, "approved", True))
            if not approved:
                hard_reject = True
                hard_reason = (
                    "Underlying RiskIntelligenceEngine rejected position; "
                    "ensemble soft score cannot override hard limits. "
                    + str(getattr(eng_decision, "reason", ""))
                )
                # Escalate ensemble state if engine is in TRADING_HALT
                if engine_state == RiskState.TRADING_HALT:
                    self._state_machine.transition(
                        assessment.dimension_scores,
                        force_state=RiskState.TRADING_HALT,
                    )
                    # Rebuild assessment state reflection
                    assessment = RiskAssessment(
                        timestamp=assessment.timestamp,
                        data_version=assessment.data_version,
                        risk_model_versions=dict(assessment.risk_model_versions),
                        input_metrics=dict(assessment.input_metrics),
                        normalized_metrics=dict(assessment.normalized_metrics),
                        dimension_scores=assessment.dimension_scores,
                        overall_score=assessment.overall_score,
                        confidence=assessment.confidence,
                        disagreement=dict(assessment.disagreement),
                        risk_state=RiskState.TRADING_HALT,
                        budget_recommendation=dict(assessment.budget_recommendation),
                        max_exposure=0.0,
                        recommended_leverage=0.0,
                        reasons=list(assessment.reasons)
                        + ["Engine TRADING_HALT forced ensemble halt"],
                        missing_critical=list(assessment.missing_critical),
                        fallback_applied=assessment.fallback_applied,
                        audit={**assessment.audit, "engine_halt": True},
                    )

        extra = []
        if missing:
            extra.append(
                "validate_position: critical metrics missing at gate: " + ",".join(missing)
            )
        if forecast_confidence > 0:
            extra.append(
                f"forecast_confidence={forecast_confidence:.2f} noted; cannot override hard limits"
            )

        decision = self.decision(
            assessment=assessment,
            proposed_exposure=proposed_exposure,
            forecast_confidence=float(forecast_confidence),
            extra_reasons=extra,
            hard_reject=hard_reject,
            hard_reject_reason=hard_reason,
            engine_audit=engine_audit,
        )

        # Final exposure/leverage caps from state
        cap = state_cap(self.settings, decision.risk_state)
        max_exp = min(float(cap.max_exposure), float(self.settings.limits.max_gross_exposure))
        lev = min(float(decision.recommended_leverage), float(cap.recommended_leverage))
        if decision.maximum_permitted_exposure != max_exp or decision.recommended_leverage != lev:
            decision = EnsembleDecision(
                decision=decision.decision,
                risk_state=decision.risk_state,
                risk_score=decision.risk_score,
                risk_confidence=decision.risk_confidence,
                triggered_limits=list(decision.triggered_limits),
                reasons=list(decision.reasons),
                required_position_reduction=max(
                    float(decision.required_position_reduction), float(cap.position_reduction)
                ),
                maximum_permitted_exposure=float(max_exp),
                recommended_leverage=float(lev),
                timestamp=decision.timestamp,
                data_version=decision.data_version,
                model_versions=dict(decision.model_versions),
                audit={**decision.audit, "caps_reapplied": True},
                proposed_exposure=decision.proposed_exposure,
                forecast_confidence=decision.forecast_confidence,
            )
            self._last_decision = decision

        return decision

    def recommended_leverage(
        self,
        *,
        realized_vol: float | None = None,
        current_drawdown: float = 0.0,
        confidence: float = 0.0,
        liquidity_score: float = 1.0,
        regime: str = "normal",
        metrics: dict[str, Any] | None = None,
        assessment: RiskAssessment | None = None,
        forecast_vol: float | None = None,
    ) -> float:
        """Recommend leverage within hard state caps. Confidence cannot breach caps."""
        if assessment is None and metrics is not None:
            assessment = self.aggregate(metrics, regime=regime)
        if assessment is not None:
            cap = state_cap(self.settings, assessment.risk_state)
            hard_max = float(cap.recommended_leverage)
            # Soft vol targeting within cap
            vol = float(
                realized_vol
                if realized_vol is not None
                else (assessment.input_metrics.get("volatility") or self.settings.leverage.target_volatility)
            )
            measure = eng_recommended_leverage(
                realized_vol=max(vol, float(forecast_vol) if forecast_vol is not None else vol),
                target_vol=self.settings.leverage.target_volatility,
                current_drawdown=float(
                    current_drawdown
                    if current_drawdown
                    else assessment.input_metrics.get("drawdown", 0.0) or 0.0
                ),
                max_drawdown=self.settings.drawdown.trading_halt,
                confidence=float(confidence),
                liquidity_score=float(liquidity_score),
                regime=regime,
                base_leverage=self.settings.leverage.base_leverage,
                max_leverage=min(hard_max, self.settings.leverage.max_leverage),
                min_leverage=self.settings.leverage.min_leverage,
                confidence_cap=self.settings.leverage.confidence_cap,
            )
            return float(min(float(measure.value), hard_max))

        # No assessment — use NORMAL caps only if vol provided; still clip hard
        vol = float(realized_vol if realized_vol is not None else self.settings.leverage.target_volatility)
        measure = eng_recommended_leverage(
            realized_vol=vol,
            target_vol=self.settings.leverage.target_volatility,
            current_drawdown=float(current_drawdown),
            max_drawdown=self.settings.drawdown.trading_halt,
            confidence=float(confidence),
            liquidity_score=float(liquidity_score),
            regime=regime,
            base_leverage=self.settings.leverage.base_leverage,
            max_leverage=self.settings.leverage.max_leverage,
            min_leverage=self.settings.leverage.min_leverage,
            confidence_cap=self.settings.leverage.confidence_cap,
        )
        return float(measure.value)

    def maximum_exposure(
        self,
        *,
        metrics: dict[str, Any] | None = None,
        assessment: RiskAssessment | None = None,
        scores: RiskScore | dict[str, Any] | None = None,
        previous_state: RiskState | None = None,
    ) -> float:
        if assessment is None:
            if metrics is not None:
                assessment = self.aggregate(metrics, previous_state=previous_state)
            elif scores is not None:
                state = self.risk_state(scores, previous_state=previous_state)
                return float(state_cap(self.settings, state).max_exposure)
            else:
                return float(state_cap(self.settings, self._state_machine.current_state).max_exposure)
        return float(
            min(assessment.max_exposure, float(self.settings.limits.max_gross_exposure))
        )

    # --------------------------------------------------------------- helpers
    def calibrate(self, **kwargs: Any) -> dict[str, Any]:
        return run_calibration(settings=self.settings, **kwargs)

    def diagnostics(self) -> dict[str, Any]:
        return self._diagnostics.health(
            assessment=self._last_assessment,
            decision=self._last_decision,
            state_machine_state=self._state_machine.export_state(),
        )

    def visualize(self, assessment: RiskAssessment | None = None) -> dict[str, Any]:
        ass = assessment or self._last_assessment
        if ass is None:
            return {}
        return self._visualization.assessment(ass)

    def export_state(self) -> dict[str, Any]:
        return {
            "settings": self.settings.model_dump(),
            "state_machine": self._state_machine.export_state(),
            "last_assessment": self._last_assessment.to_dict() if self._last_assessment else None,
            "last_decision": self._last_decision.to_dict() if self._last_decision else None,
            "audit_log_len": len(self._audit_log),
            "ensemble_version": self.settings.ensemble_version,
        }

    def import_state(self, payload: dict[str, Any]) -> RiskIntelligenceEnsemble:
        if not payload:
            return self
        sm = payload.get("state_machine")
        if isinstance(sm, dict):
            self._state_machine.import_state(sm)
        return self

    def to_json(self, obj: Any | None = None) -> str:
        return self._serializer.to_json(obj if obj is not None else self.export_state())
