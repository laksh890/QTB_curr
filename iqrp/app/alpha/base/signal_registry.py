"""In-memory trial / experiment registry for alpha research.

CRITICAL:
- Rejected experiments are preserved (never silently deleted).
- Status transitions are auditable (from→to, reason, timestamp).
- Statistical significance alone ≠ alpha; Historical Sharpe alone cannot approve.
- SignalDefinition.economic_hypothesis must be tracked on every experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any
from uuid import uuid4

from iqrp.app.alpha.base.alpha_signal import AlphaSignal
from iqrp.app.alpha.base.signal_definition import SignalDefinition
from iqrp.app.alpha.base.signal_metadata import SignalMetadata
from iqrp.app.alpha.base.signal_result import (
    SignalResearchReport,
    SignalStatus,
    StatusTransition,
    validate_transition,
)


@dataclass(slots=True)
class ExperimentRecord:
    """A single registered trial / experiment (including rejected ones)."""

    experiment_id: str
    definition: SignalDefinition
    status: SignalStatus
    metadata: SignalMetadata
    signal: AlphaSignal | None = None
    report: SignalResearchReport | None = None
    transitions: list[StatusTransition] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    rejected: bool = False
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "definition": self.definition.to_dict(),
            "status": self.status.value,
            "metadata": self.metadata.to_dict(),
            "signal": None if self.signal is None else self.signal.to_dict(),
            "report": None if self.report is None else self.report.to_dict(),
            "transitions": [t.to_dict() for t in self.transitions],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "rejected": self.rejected,
            "tags": list(self.tags),
        }


class SignalRegistry:
    """Thread-safe in-memory registry that preserves rejected experiments."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._experiments: dict[str, ExperimentRecord] = {}
        self._by_definition: dict[str, list[str]] = {}

    def register(
        self,
        definition: SignalDefinition,
        *,
        signal: AlphaSignal | None = None,
        metadata: SignalMetadata | None = None,
        status: SignalStatus = SignalStatus.CANDIDATE,
        experiment_id: str | None = None,
        tags: tuple[str, ...] = (),
        actor: str = "system",
        reason: str = "initial registration",
    ) -> ExperimentRecord:
        # Candidates may register with a thin/empty hypothesis; APPROVED still
        # requires a substantive economic_hypothesis (see transition()).
        eid = experiment_id or str(uuid4())
        meta = metadata or SignalMetadata(
            signal_name=definition.name,
            version=definition.version,
            universe=definition.universe,
            frequency=definition.frequency,
            owner=definition.owner,
            economic_hypothesis=definition.economic_hypothesis,
            tags=definition.tags,
        )
        now = datetime.now(UTC)
        transition = StatusTransition(
            from_status=SignalStatus.CANDIDATE,
            to_status=status,
            reason=reason,
            timestamp=now,
            actor=actor,
        )
        # Initial registration: if starting as CANDIDATE, record self-transition for audit
        if status == SignalStatus.CANDIDATE:
            transition = StatusTransition(
                from_status=SignalStatus.CANDIDATE,
                to_status=SignalStatus.CANDIDATE,
                reason=reason,
                timestamp=now,
                actor=actor,
                extras={"event": "register"},
            )
        elif status != SignalStatus.CANDIDATE:
            validate_transition(SignalStatus.CANDIDATE, status)

        record = ExperimentRecord(
            experiment_id=eid,
            definition=definition,
            status=status,
            metadata=meta,
            signal=signal,
            transitions=[transition],
            created_at=now,
            updated_at=now,
            rejected=status == SignalStatus.REJECTED,
            tags=tags or definition.tags,
        )
        with self._lock:
            if eid in self._experiments:
                raise KeyError(f"experiment_id already exists: {eid}")
            self._experiments[eid] = record
            self._by_definition.setdefault(definition.definition_id, []).append(eid)
        return record

    def get(self, experiment_id: str) -> ExperimentRecord:
        with self._lock:
            if experiment_id not in self._experiments:
                raise KeyError(f"Unknown experiment_id: {experiment_id}")
            return self._experiments[experiment_id]

    def transition(
        self,
        experiment_id: str,
        to_status: SignalStatus,
        *,
        reason: str,
        actor: str = "system",
        extras: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        if not reason.strip():
            raise ValueError("transition reason is required for auditability")
        with self._lock:
            record = self.get(experiment_id)
            validate_transition(record.status, to_status)
            if to_status == SignalStatus.APPROVED:
                # Historical Sharpe alone cannot approve.
                hyp = record.definition.economic_hypothesis.strip()
                if not hyp:
                    raise ValueError(
                        "Cannot APPROVE without economic_hypothesis. "
                        "Historical Sharpe alone cannot approve."
                    )
                if len(hyp) < 20:
                    raise ValueError(
                        "economic_hypothesis too thin for APPROVED status; "
                        "provide a substantive economic rationale."
                    )
            now = datetime.now(UTC)
            tr = StatusTransition(
                from_status=record.status,
                to_status=to_status,
                reason=reason,
                timestamp=now,
                actor=actor,
                extras=dict(extras or {}),
            )
            record.transitions.append(tr)
            record.status = to_status
            record.updated_at = now
            if to_status == SignalStatus.REJECTED:
                record.rejected = True
            return record

    def attach_signal(self, experiment_id: str, signal: AlphaSignal) -> ExperimentRecord:
        with self._lock:
            record = self.get(experiment_id)
            record.signal = signal
            record.updated_at = datetime.now(UTC)
            return record

    def attach_report(
        self, experiment_id: str, report: SignalResearchReport
    ) -> ExperimentRecord:
        with self._lock:
            record = self.get(experiment_id)
            record.report = report
            record.updated_at = datetime.now(UTC)
            return record

    def list_experiments(
        self,
        *,
        status: SignalStatus | None = None,
        include_rejected: bool = True,
        definition_id: str | None = None,
    ) -> list[ExperimentRecord]:
        with self._lock:
            records = list(self._experiments.values())
        if definition_id is not None:
            ids = set(self._by_definition.get(definition_id, []))
            records = [r for r in records if r.experiment_id in ids]
        if status is not None:
            records = [r for r in records if r.status == status]
        if not include_rejected:
            records = [r for r in records if not r.rejected]
        return records

    def rejected_experiments(self) -> list[ExperimentRecord]:
        """Return all rejected experiments (preserved for audit / learning)."""
        return self.list_experiments(status=SignalStatus.REJECTED, include_rejected=True)

    def audit_trail(self, experiment_id: str) -> list[StatusTransition]:
        return list(self.get(experiment_id).transitions)

    def clear(self) -> None:
        """Clear registry (test helper). Rejected experiments are wiped only on explicit clear."""
        with self._lock:
            self._experiments.clear()
            self._by_definition.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._experiments)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "experiments": {eid: rec.to_dict() for eid, rec in self._experiments.items()},
                "n_experiments": len(self._experiments),
                "n_rejected": sum(1 for r in self._experiments.values() if r.rejected),
            }


# Module-level default registry instance
_DEFAULT_REGISTRY = SignalRegistry()


def get_default_registry() -> SignalRegistry:
    return _DEFAULT_REGISTRY
