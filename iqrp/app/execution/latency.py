"""Execution latency tracking: decision → submit → ack → fill.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Timestamps use only observed event times (no future information).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _ms(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return max((end - start).total_seconds() * 1000.0, 0.0)


@dataclass
class LatencyRecord:
    """Per-order latency timeline."""

    order_id: str
    record_id: str = field(default_factory=lambda: f"lat_{uuid4().hex[:12]}")
    decision_at: datetime | None = None
    submit_at: datetime | None = None
    ack_at: datetime | None = None
    fill_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def decision_to_submit_ms(self) -> float | None:
        return _ms(self.decision_at, self.submit_at)

    @property
    def submit_to_ack_ms(self) -> float | None:
        return _ms(self.submit_at, self.ack_at)

    @property
    def ack_to_fill_ms(self) -> float | None:
        return _ms(self.ack_at, self.fill_at)

    @property
    def decision_to_fill_ms(self) -> float | None:
        return _ms(self.decision_at, self.fill_at)

    @property
    def end_to_end_ms(self) -> float | None:
        return self.decision_to_fill_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "order_id": self.order_id,
            "decision_at": self.decision_at.isoformat() if self.decision_at else None,
            "submit_at": self.submit_at.isoformat() if self.submit_at else None,
            "ack_at": self.ack_at.isoformat() if self.ack_at else None,
            "fill_at": self.fill_at.isoformat() if self.fill_at else None,
            "decision_to_submit_ms": self.decision_to_submit_ms,
            "submit_to_ack_ms": self.submit_to_ack_ms,
            "ack_to_fill_ms": self.ack_to_fill_ms,
            "decision_to_fill_ms": self.decision_to_fill_ms,
            "end_to_end_ms": self.end_to_end_ms,
            "metadata": dict(self.metadata),
        }


class LatencyTracker:
    """Track and aggregate execution latency metrics."""

    def __init__(self) -> None:
        self._records: dict[str, LatencyRecord] = {}

    def start(self, order_id: str, *, at: datetime | str | None = None) -> LatencyRecord:
        rec = LatencyRecord(order_id=order_id, decision_at=_parse_ts(at) or _utc_now())
        self._records[order_id] = rec
        return rec

    def mark_decision(self, order_id: str, *, at: datetime | str | None = None) -> LatencyRecord:
        rec = self._ensure(order_id)
        rec.decision_at = _parse_ts(at) or _utc_now()
        return rec

    def mark_submit(self, order_id: str, *, at: datetime | str | None = None) -> LatencyRecord:
        rec = self._ensure(order_id)
        if rec.decision_at is None:
            rec.decision_at = _parse_ts(at) or _utc_now()
        rec.submit_at = _parse_ts(at) or _utc_now()
        return rec

    def mark_ack(self, order_id: str, *, at: datetime | str | None = None) -> LatencyRecord:
        rec = self._ensure(order_id)
        rec.ack_at = _parse_ts(at) or _utc_now()
        return rec

    def mark_fill(self, order_id: str, *, at: datetime | str | None = None) -> LatencyRecord:
        rec = self._ensure(order_id)
        rec.fill_at = _parse_ts(at) or _utc_now()
        return rec

    def get(self, order_id: str) -> LatencyRecord | None:
        return self._records.get(order_id)

    def summary(self, order_ids: list[str] | None = None) -> dict[str, Any]:
        ids = order_ids or list(self._records)
        records = [self._records[i] for i in ids if i in self._records]
        if not records:
            return {
                "n": 0,
                "avg_decision_to_submit_ms": None,
                "avg_submit_to_ack_ms": None,
                "avg_ack_to_fill_ms": None,
                "avg_end_to_end_ms": None,
                "p50_end_to_end_ms": None,
                "p95_end_to_end_ms": None,
                "records": [],
            }

        def _avg(vals: list[float]) -> float | None:
            return float(sum(vals) / len(vals)) if vals else None

        def _pct(vals: list[float], p: float) -> float | None:
            if not vals:
                return None
            s = sorted(vals)
            idx = min(max(int(round((p / 100.0) * (len(s) - 1))), 0), len(s) - 1)
            return float(s[idx])

        d2s = [r.decision_to_submit_ms for r in records if r.decision_to_submit_ms is not None]
        s2a = [r.submit_to_ack_ms for r in records if r.submit_to_ack_ms is not None]
        a2f = [r.ack_to_fill_ms for r in records if r.ack_to_fill_ms is not None]
        e2e = [r.end_to_end_ms for r in records if r.end_to_end_ms is not None]
        return {
            "n": len(records),
            "avg_decision_to_submit_ms": _avg([float(v) for v in d2s]),
            "avg_submit_to_ack_ms": _avg([float(v) for v in s2a]),
            "avg_ack_to_fill_ms": _avg([float(v) for v in a2f]),
            "avg_end_to_end_ms": _avg([float(v) for v in e2e]),
            "p50_end_to_end_ms": _pct([float(v) for v in e2e], 50),
            "p95_end_to_end_ms": _pct([float(v) for v in e2e], 95),
            "records": [r.to_dict() for r in records],
        }

    def to_dict(self) -> dict[str, Any]:
        return {"records": {k: v.to_dict() for k, v in self._records.items()}}

    def _ensure(self, order_id: str) -> LatencyRecord:
        if order_id not in self._records:
            self._records[order_id] = LatencyRecord(order_id=order_id)
        return self._records[order_id]


__all__ = ["LatencyRecord", "LatencyTracker"]
