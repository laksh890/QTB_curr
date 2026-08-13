"""Position reconciliation: expected vs executed vs broker.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Alerts on material diffs; never silently invent fills or future positions.
- No future information — reconciliation uses only observed quantities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DiffSeverity(str, Enum):
    NONE = "NONE"
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(slots=True)
class PositionSnapshot:
    instrument: str
    quantity: float
    source: str  # expected | executed | broker
    timestamp: str = field(default_factory=_utc_now)
    notional: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReconciliationAlert:
    instrument: str
    severity: DiffSeverity
    message: str
    expected: float
    executed: float
    broker: float
    expected_vs_executed: float
    executed_vs_broker: float
    expected_vs_broker: float
    timestamp: str = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "severity": self.severity.value,
            "message": self.message,
            "expected": self.expected,
            "executed": self.executed,
            "broker": self.broker,
            "expected_vs_executed": self.expected_vs_executed,
            "executed_vs_broker": self.executed_vs_broker,
            "expected_vs_broker": self.expected_vs_broker,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ReconciliationResult:
    matched: bool
    alerts: list[ReconciliationAlert] = field(default_factory=list)
    per_instrument: dict[str, dict[str, float]] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "alerts": [a.to_dict() for a in self.alerts],
            "per_instrument": dict(self.per_instrument),
            "timestamp": self.timestamp,
        }


@dataclass
class PositionReconciler:
    """Compare expected, executed, and broker positions; alert on diffs."""

    qty_tolerance: float = 0.0
    notional_tolerance: float = 0.01
    alert_on_diff: bool = True
    _alerts: list[ReconciliationAlert] = field(default_factory=list)

    def reconcile(
        self,
        *,
        expected: dict[str, float],
        executed: dict[str, float],
        broker: dict[str, float],
    ) -> ReconciliationResult:
        instruments = sorted(set(expected) | set(executed) | set(broker))
        alerts: list[ReconciliationAlert] = []
        per: dict[str, dict[str, float]] = {}

        for inst in instruments:
            key = str(inst).upper()
            exp = float(expected.get(inst, expected.get(key, 0.0)))
            exe = float(executed.get(inst, executed.get(key, 0.0)))
            brk = float(broker.get(inst, broker.get(key, 0.0)))
            d_ee = exe - exp
            d_eb = brk - exe
            d_xb = brk - exp
            per[key] = {
                "expected": exp,
                "executed": exe,
                "broker": brk,
                "expected_vs_executed": d_ee,
                "executed_vs_broker": d_eb,
                "expected_vs_broker": d_xb,
            }

            abs_diffs = (abs(d_ee), abs(d_eb), abs(d_xb))
            if max(abs_diffs) <= self.qty_tolerance + 1e-12:
                continue

            severity = DiffSeverity.WARNING
            if max(abs_diffs) > max(self.qty_tolerance, 1.0) * 10:
                severity = DiffSeverity.CRITICAL
            elif max(abs_diffs) <= self.qty_tolerance + 1.0:
                severity = DiffSeverity.INFO

            msg = (
                f"{key}: expected={exp}, executed={exe}, broker={brk} "
                f"(Δee={d_ee}, Δeb={d_eb}, Δxb={d_xb})"
            )
            alert = ReconciliationAlert(
                instrument=key,
                severity=severity,
                message=msg,
                expected=exp,
                executed=exe,
                broker=brk,
                expected_vs_executed=d_ee,
                executed_vs_broker=d_eb,
                expected_vs_broker=d_xb,
            )
            if self.alert_on_diff:
                alerts.append(alert)
                self._alerts.append(alert)

        return ReconciliationResult(
            matched=len(alerts) == 0,
            alerts=alerts,
            per_instrument=per,
        )

    @property
    def alerts(self) -> list[ReconciliationAlert]:
        return list(self._alerts)
