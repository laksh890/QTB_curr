"""Append-only audit log for order lifecycle events.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Audit records are append-only; no mutation or deletion of history.
- No future information may be recorded as present fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class AuditEntry:
    """Single immutable audit record."""

    event_type: str
    message: str
    order_id: str | None = None
    actor: str = "system"
    timestamp: str = field(default_factory=_utc_now)
    entry_id: str = field(default_factory=lambda: str(uuid4()))
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "event_type": self.event_type,
            "message": self.message,
            "order_id": self.order_id,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "details": dict(self.details),
        }


@dataclass
class AuditLog:
    """Append-only audit trail.

    Entries are never modified or removed after append. Consumers may filter
    but must not rewrite history.
    """

    _entries: list[AuditEntry] = field(default_factory=list, repr=False)

    def append(
        self,
        event_type: str,
        message: str,
        *,
        order_id: str | None = None,
        actor: str = "system",
        details: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            event_type=event_type,
            message=message,
            order_id=order_id,
            actor=actor,
            timestamp=timestamp or _utc_now(),
            details=dict(details or {}),
        )
        self._entries.append(entry)
        return entry

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    @property
    def entries(self) -> tuple[AuditEntry, ...]:
        """Immutable view of all entries."""
        return tuple(self._entries)

    def for_order(self, order_id: str) -> list[AuditEntry]:
        return [e for e in self._entries if e.order_id == order_id]

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    def clear_for_tests_only(self) -> None:
        """Dangerous: only for unit-test isolation. Never use in production."""
        self._entries.clear()
