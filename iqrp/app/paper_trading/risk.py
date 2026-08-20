"""Paper-trading kill switches and risk halt state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KillSwitchState:
    halted: bool = False
    reasons: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def trip(self, reason: str, *, meta: dict[str, Any] | None = None) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)
        self.halted = True
        self.events.append({"reason": reason, "meta": meta or {}})

    def clear_for_test(self) -> None:
        """Only for failure-injection recovery tests."""
        self.halted = False
        self.reasons = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "halted": self.halted,
            "reasons": list(self.reasons),
            "n_events": len(self.events),
            "events": self.events[-50:],
        }


@dataclass
class PaperRiskLimits:
    max_position: float = 0.20
    max_gross: float = 1.0
    max_net: float = 1.0
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.25
    max_turnover_per_bar: float = 0.50
    stale_bars_threshold: int = 5


def check_risk(
    *,
    limits: PaperRiskLimits,
    kill: KillSwitchState,
    target_weight: float,
    current_weight: float,
    equity: float,
    peak_equity: float,
    day_start_equity: float,
    stale_bars: int = 0,
    model_failed: bool = False,
    recon_failed: bool = False,
    exec_failed: bool = False,
) -> tuple[float, list[str]]:
    """Return (possibly reduced target weight, rejection reasons)."""
    reasons: list[str] = []
    if kill.halted:
        return 0.0, ["KILL_SWITCH_ACTIVE"] + list(kill.reasons)

    if model_failed:
        kill.trip("MODEL_FAILURE")
        return 0.0, ["MODEL_FAILURE"]
    if recon_failed:
        kill.trip("RECONCILIATION_FAILURE")
        return 0.0, ["RECONCILIATION_FAILURE"]
    if exec_failed:
        kill.trip("EXECUTION_FAILURE")
        return 0.0, ["EXECUTION_FAILURE"]
    if stale_bars >= limits.stale_bars_threshold:
        kill.trip("STALE_DATA", meta={"stale_bars": stale_bars})
        return 0.0, ["STALE_DATA"]

    if peak_equity > 0:
        dd = (peak_equity - equity) / peak_equity
        if dd >= limits.max_drawdown:
            kill.trip("MAX_DRAWDOWN", meta={"dd": dd})
            return 0.0, ["MAX_DRAWDOWN"]

    if day_start_equity > 0:
        day_loss = (day_start_equity - equity) / day_start_equity
        if day_loss >= limits.max_daily_loss:
            kill.trip("MAX_DAILY_LOSS", meta={"day_loss": day_loss})
            return 0.0, ["MAX_DAILY_LOSS"]

    import numpy as np

    tw = float(np.clip(target_weight, -limits.max_position, limits.max_position))
    if abs(tw) > limits.max_gross + 1e-12:
        tw = float(np.sign(tw) * limits.max_gross)
        reasons.append("MAX_GROSS_CLIP")
    if abs(tw) > limits.max_net + 1e-12:
        tw = float(np.sign(tw) * limits.max_net)
        reasons.append("MAX_NET_CLIP")

    turnover = abs(tw - current_weight)
    if turnover > limits.max_turnover_per_bar + 1e-12:
        direction = 1.0 if tw > current_weight else -1.0
        tw = current_weight + direction * limits.max_turnover_per_bar
        reasons.append("MAX_TURNOVER_CLIP")

    if abs(target_weight) > limits.max_position + 1e-12:
        reasons.append("MAX_POSITION_CLIP")

    return float(tw), reasons


__all__ = ["KillSwitchState", "PaperRiskLimits", "check_risk"]
