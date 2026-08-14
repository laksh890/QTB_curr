"""Risk limit definitions and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from iqrp.app.risk.base.risk_measure import LimitBreach, LimitSeverity


@dataclass(slots=True)
class RiskLimit:
    name: str
    threshold: float
    severity: LimitSeverity = LimitSeverity.HARD
    scope: str = "portfolio"
    direction: str = "max"  # max: breach if observed > threshold; min: if observed < threshold
    metadata: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, observed: float) -> LimitBreach | None:
        breached = (
            observed > self.threshold if self.direction == "max" else observed < self.threshold
        )
        if not breached:
            return None
        return LimitBreach(
            limit_name=self.name,
            severity=self.severity,
            observed=float(observed),
            threshold=float(self.threshold),
            reason=f"{self.name}: observed={observed:.6g} vs threshold={self.threshold:.6g} ({self.severity.value})",
            scope=self.scope,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "scope": self.scope,
            "direction": self.direction,
            "metadata": dict(self.metadata),
        }


def evaluate_limits(limits: list[RiskLimit], values: dict[str, float]) -> list[LimitBreach]:
    out: list[LimitBreach] = []
    for lim in limits:
        if lim.name in values:
            b = lim.evaluate(float(values[lim.name]))
            if b is not None:
                out.append(b)
    return out
