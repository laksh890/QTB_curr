"""Deterministic RiskState machine with hysteresis and multi-dimension confirmation.

Transitions: NORMAL ↔ CAUTION ↔ REDUCED_RISK ↔ CAPITAL_PRESERVATION ↔ TRADING_HALT

Rules:
- Escalation uses max of confirmed multi-metric signals vs thresholds.
- Recovery requires N consecutive confirmations below recovery thresholds.
- Single noisy metric must NOT trigger TRADING_HALT unless hard_halt_on_single.
- Counters persist on the ensemble / machine instance.
"""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.base import RiskState
from iqrp.app.risk.ensemble.config import EnsembleSettings
from iqrp.app.risk.ensemble.types import RiskScore

_STATE_ORDER: tuple[RiskState, ...] = (
    RiskState.NORMAL,
    RiskState.CAUTION,
    RiskState.REDUCED_RISK,
    RiskState.CAPITAL_PRESERVATION,
    RiskState.TRADING_HALT,
)

_STATE_RANK = {s: i for i, s in enumerate(_STATE_ORDER)}


def _score_to_candidate(overall: float, settings: EnsembleSettings) -> RiskState:
    t = settings.state_thresholds
    if overall >= t.trading_halt:
        return RiskState.TRADING_HALT
    if overall >= t.capital_preservation:
        return RiskState.CAPITAL_PRESERVATION
    if overall >= t.reduced_risk:
        return RiskState.REDUCED_RISK
    if overall >= t.caution:
        return RiskState.CAUTION
    return RiskState.NORMAL


def _recovery_ceiling(overall: float, settings: EnsembleSettings) -> RiskState:
    """Highest state still justified under recovery (hysteresis) thresholds."""
    r = settings.recovery_thresholds
    if overall >= r.trading_halt:
        return RiskState.TRADING_HALT
    if overall >= r.capital_preservation:
        return RiskState.CAPITAL_PRESERVATION
    if overall >= r.reduced_risk:
        return RiskState.REDUCED_RISK
    if overall >= r.caution:
        return RiskState.CAUTION
    return RiskState.NORMAL


def _confirmed_hot_dimensions(scores: RiskScore, threshold: float) -> list[str]:
    return [name for name, val in scores.dimension_map().items() if float(val) >= threshold]


def _raw_escalation_target(
    scores: RiskScore,
    settings: EnsembleSettings,
) -> tuple[RiskState, dict[str, Any]]:
    """Determine raw escalation target with multi-dimension halt confirmation."""
    overall = float(scores.overall)
    candidate = _score_to_candidate(overall, settings)
    hot = _confirmed_hot_dimensions(
        scores, float(settings.hysteresis.dimension_confirmation_threshold)
    )
    meta: dict[str, Any] = {
        "overall": overall,
        "score_candidate": candidate.value,
        "hot_dimensions": hot,
        "hard_halt_on_single": bool(settings.hard_halt_on_single),
    }

    if candidate == RiskState.TRADING_HALT:
        min_dims = max(int(settings.min_dimensions_for_halt), 1)
        if settings.hard_halt_on_single:
            meta["halt_confirmation"] = "hard_halt_on_single"
            return RiskState.TRADING_HALT, meta
        if len(hot) >= min_dims or overall >= settings.state_thresholds.trading_halt and len(hot) >= 2:
            meta["halt_confirmation"] = "multi_dimension"
            return RiskState.TRADING_HALT, meta
        # Single noisy metric / insufficient confirmation → cap at CAPITAL_PRESERVATION
        meta["halt_confirmation"] = "blocked_insufficient_dimensions"
        meta["downgraded_to"] = RiskState.CAPITAL_PRESERVATION.value
        return RiskState.CAPITAL_PRESERVATION, meta

    return candidate, meta


class EnsembleStateMachine:
    """Stateful hysteresis machine; counters live on this instance (and ensemble)."""

    def __init__(self, settings: EnsembleSettings) -> None:
        self.settings = settings
        self.current_state: RiskState = RiskState.NORMAL
        self._escalation_streak: int = 0
        self._recovery_streak: int = 0
        self._pending_escalation: RiskState | None = None
        self._pending_recovery: RiskState | None = None
        self._history: list[dict[str, Any]] = []

    def reset(self, state: RiskState = RiskState.NORMAL) -> None:
        self.current_state = state
        self._escalation_streak = 0
        self._recovery_streak = 0
        self._pending_escalation = None
        self._pending_recovery = None

    def export_state(self) -> dict[str, Any]:
        return {
            "current_state": self.current_state.value,
            "escalation_streak": self._escalation_streak,
            "recovery_streak": self._recovery_streak,
            "pending_escalation": self._pending_escalation.value if self._pending_escalation else None,
            "pending_recovery": self._pending_recovery.value if self._pending_recovery else None,
        }

    def import_state(self, payload: dict[str, Any]) -> None:
        if not payload:
            return
        if "current_state" in payload and payload["current_state"]:
            self.current_state = RiskState(str(payload["current_state"]))
        self._escalation_streak = int(payload.get("escalation_streak", 0) or 0)
        self._recovery_streak = int(payload.get("recovery_streak", 0) or 0)
        pe = payload.get("pending_escalation")
        pr = payload.get("pending_recovery")
        self._pending_escalation = RiskState(str(pe)) if pe else None
        self._pending_recovery = RiskState(str(pr)) if pr else None

    def transition(
        self,
        scores: RiskScore | dict[str, Any],
        *,
        previous_state: RiskState | None = None,
        force_state: RiskState | None = None,
    ) -> RiskState:
        if isinstance(scores, dict):
            scores = RiskScore.from_dict(scores)

        if previous_state is not None:
            self.current_state = previous_state

        if force_state is not None:
            self.current_state = force_state
            self._escalation_streak = 0
            self._recovery_streak = 0
            self._pending_escalation = None
            self._pending_recovery = None
            self._history.append(
                {"forced": True, "state": force_state.value, "reason": "force_state"}
            )
            return self.current_state

        target, meta = _raw_escalation_target(scores, self.settings)
        prev = self.current_state
        prev_rank = _STATE_RANK[prev]
        target_rank = _STATE_RANK[target]

        esc_needed = max(int(self.settings.hysteresis.escalation_confirmations), 1)
        rec_needed = max(int(self.settings.hysteresis.recovery_confirmations), 1)

        if target_rank > prev_rank:
            # Escalation path
            self._recovery_streak = 0
            self._pending_recovery = None
            if self._pending_escalation != target:
                self._pending_escalation = target
                self._escalation_streak = 1
            else:
                self._escalation_streak += 1
            if self._escalation_streak >= esc_needed:
                self.current_state = target
                self._escalation_streak = 0
                self._pending_escalation = None
                action = "escalated"
            else:
                action = "escalation_pending"
        elif target_rank < prev_rank:
            # Recovery: must clear recovery thresholds for the step down
            recovery_cap = _recovery_ceiling(float(scores.overall), self.settings)
            # May only recover to recovery_cap (or lower)
            desired = recovery_cap if _STATE_RANK[recovery_cap] < prev_rank else prev
            # Step down at most one level per confirmed recovery cycle for stability
            step_target = _STATE_ORDER[prev_rank - 1] if prev_rank > 0 else RiskState.NORMAL
            if _STATE_RANK[desired] > _STATE_RANK[step_target]:
                step_target = desired  # still higher than one-step; wait
                # If desired is not below previous, no recovery yet
                if _STATE_RANK[desired] >= prev_rank:
                    self._recovery_streak = 0
                    self._pending_recovery = None
                    action = "recovery_blocked_hysteresis"
                    self._history.append(
                        {
                            "action": action,
                            "previous": prev.value,
                            "target": target.value,
                            "state": self.current_state.value,
                            **meta,
                        }
                    )
                    return self.current_state

            self._escalation_streak = 0
            self._pending_escalation = None
            if self._pending_recovery != step_target:
                self._pending_recovery = step_target
                self._recovery_streak = 1
            else:
                self._recovery_streak += 1
            if self._recovery_streak >= rec_needed:
                self.current_state = step_target
                self._recovery_streak = 0
                self._pending_recovery = None
                action = "recovered"
            else:
                action = "recovery_pending"
        else:
            self._escalation_streak = 0
            self._recovery_streak = 0
            self._pending_escalation = None
            self._pending_recovery = None
            action = "hold"

        self._history.append(
            {
                "action": action,
                "previous": prev.value,
                "target": target.value,
                "state": self.current_state.value,
                "escalation_streak": self._escalation_streak,
                "recovery_streak": self._recovery_streak,
                **meta,
            }
        )
        return self.current_state

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)
