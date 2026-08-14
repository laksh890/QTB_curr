"""Retraining schedule triggers: time, performance, drift, regime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

TriggerKind = Literal["time", "performance", "drift", "regime", "composite"]


@dataclass
class TriggerDecision:
    should_retrain: bool
    kind: TriggerKind | str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_retrain": bool(self.should_retrain),
            "kind": str(self.kind),
            "reason": str(self.reason),
            "details": dict(self.details),
        }


class RetrainTrigger(ABC):
    """Base class for retrain triggers."""

    kind: TriggerKind = "time"

    @abstractmethod
    def evaluate(self, context: dict[str, Any]) -> TriggerDecision:
        """Return whether a retrain should fire given ``context``."""


@dataclass
class TimeTrigger(RetrainTrigger):
    """Fire every ``every`` bars since last retrain (or from origin)."""

    every: int = 20
    kind: TriggerKind = "time"

    def evaluate(self, context: dict[str, Any]) -> TriggerDecision:
        t = int(context.get("t", context.get("index", 0)))
        last = int(context.get("last_retrain_t", context.get("trained_through", -1)))
        every = max(int(self.every), 1)
        elapsed = t - last
        fire = elapsed >= every and t > last
        return TriggerDecision(
            should_retrain=fire,
            kind=self.kind,
            reason=f"elapsed={elapsed} every={every}" if fire else "interval_not_reached",
            details={"t": t, "last_retrain_t": last, "every": every, "elapsed": elapsed},
        )


@dataclass
class PerformanceTrigger(RetrainTrigger):
    """Fire when a rolling performance metric falls below a floor."""

    metric: str = "sharpe"
    min_value: float = 0.0
    lookback: int = 20
    kind: TriggerKind = "performance"

    def evaluate(self, context: dict[str, Any]) -> TriggerDecision:
        series = context.get("performance")
        metric_val = context.get(self.metric)
        if metric_val is None and series is not None:
            arr = np.asarray(series, dtype=np.float64).reshape(-1)
            if arr.size:
                lb = max(int(self.lookback), 1)
                window = arr[-lb:]
                # Treat series as returns → simple Sharpe proxy if metric is sharpe.
                if self.metric == "sharpe":
                    sd = float(np.std(window, ddof=1)) if window.size > 1 else 0.0
                    metric_val = float(np.mean(window) / sd * np.sqrt(252.0)) if sd > 1e-15 else 0.0
                else:
                    metric_val = float(np.mean(window))
        if metric_val is None:
            return TriggerDecision(False, self.kind, "metric_unavailable", {"metric": self.metric})
        val = float(metric_val)
        fire = val < float(self.min_value)
        return TriggerDecision(
            should_retrain=fire,
            kind=self.kind,
            reason=f"{self.metric}={val:.6g} < {self.min_value}" if fire else "ok",
            details={"metric": self.metric, "value": val, "min_value": float(self.min_value)},
        )


@dataclass
class DriftTrigger(RetrainTrigger):
    """Fire when a drift score exceeds a threshold."""

    threshold: float = 0.25
    score_key: str = "drift_score"
    kind: TriggerKind = "drift"

    def evaluate(self, context: dict[str, Any]) -> TriggerDecision:
        score = context.get(self.score_key)
        if score is None:
            # Optional: compute PSI-like score from ref vs cur feature means.
            ref = context.get("feature_ref")
            cur = context.get("feature_cur")
            if ref is not None and cur is not None:
                r = np.asarray(ref, dtype=np.float64)
                c = np.asarray(cur, dtype=np.float64)
                if r.size and c.size:
                    # Mean absolute standardized shift.
                    rs = float(np.std(r)) if r.size > 1 else 1.0
                    score = abs(float(np.mean(c)) - float(np.mean(r))) / (rs + 1e-12)
        if score is None:
            return TriggerDecision(False, self.kind, "score_unavailable", {})
        val = float(score)
        fire = val >= float(self.threshold)
        return TriggerDecision(
            should_retrain=fire,
            kind=self.kind,
            reason=f"drift={val:.6g} >= {self.threshold}" if fire else "ok",
            details={"drift_score": val, "threshold": float(self.threshold)},
        )


@dataclass
class RegimeTrigger(RetrainTrigger):
    """Fire when the detected regime label changes."""

    regime_key: str = "regime"
    kind: TriggerKind = "regime"

    def evaluate(self, context: dict[str, Any]) -> TriggerDecision:
        current = context.get(self.regime_key)
        previous = context.get("previous_regime", context.get("last_regime"))
        if current is None:
            return TriggerDecision(False, self.kind, "regime_unavailable", {})
        fire = previous is not None and current != previous
        return TriggerDecision(
            should_retrain=fire,
            kind=self.kind,
            reason=f"regime {previous!r} -> {current!r}" if fire else "unchanged",
            details={"regime": current, "previous_regime": previous},
        )


@dataclass
class CompositeTrigger(RetrainTrigger):
    """Combine triggers with OR (any) or AND (all) logic."""

    triggers: list[RetrainTrigger] = field(default_factory=list)
    combine: Literal["any", "all"] = "any"
    kind: TriggerKind = "composite"

    def evaluate(self, context: dict[str, Any]) -> TriggerDecision:
        if not self.triggers:
            return TriggerDecision(False, self.kind, "no_triggers", {})
        decisions = [t.evaluate(context) for t in self.triggers]
        if self.combine == "all":
            fire = all(d.should_retrain for d in decisions)
        else:
            fire = any(d.should_retrain for d in decisions)
        reasons = [d.reason for d in decisions if d.should_retrain] if fire else ["idle"]
        return TriggerDecision(
            should_retrain=fire,
            kind=self.kind,
            reason="; ".join(reasons),
            details={"combine": self.combine, "children": [d.to_dict() for d in decisions]},
        )


@dataclass
class RetrainSchedule:
    """Schedule wrapper around one or more triggers."""

    trigger: RetrainTrigger
    min_bars_between: int = 0
    max_retrains: int | None = None

    def __init__(
        self,
        trigger: RetrainTrigger | None = None,
        *,
        every: int | None = None,
        min_bars_between: int = 0,
        max_retrains: int | None = None,
        triggers: list[RetrainTrigger] | None = None,
        combine: Literal["any", "all"] = "any",
    ) -> None:
        if trigger is not None:
            self.trigger = trigger
        elif triggers:
            self.trigger = CompositeTrigger(triggers=list(triggers), combine=combine)
        elif every is not None:
            self.trigger = TimeTrigger(every=int(every))
        else:
            self.trigger = TimeTrigger(every=20)
        self.min_bars_between = max(int(min_bars_between), 0)
        self.max_retrains = max_retrains
        self._retrain_count = 0

    def should_retrain(self, context: dict[str, Any]) -> TriggerDecision:
        if self.max_retrains is not None and self._retrain_count >= int(self.max_retrains):
            return TriggerDecision(
                False,
                getattr(self.trigger, "kind", "composite"),
                "max_retrains_reached",
                {"max_retrains": self.max_retrains, "count": self._retrain_count},
            )
        t = int(context.get("t", context.get("index", 0)))
        last = int(context.get("last_retrain_t", -(10**18)))
        if self.min_bars_between > 0 and (t - last) < self.min_bars_between:
            return TriggerDecision(
                False,
                getattr(self.trigger, "kind", "time"),
                "min_bars_between",
                {"t": t, "last_retrain_t": last, "min_bars_between": self.min_bars_between},
            )
        decision = self.trigger.evaluate(context)
        return decision

    def record_retrain(self) -> None:
        self._retrain_count += 1

    @property
    def retrain_count(self) -> int:
        return self._retrain_count
