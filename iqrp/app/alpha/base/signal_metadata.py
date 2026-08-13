"""Runtime / research metadata attached to alpha candidates.

CRITICAL:
- Metadata documents research context; it does not approve alpha.
- Historical Sharpe alone cannot approve.
- Must preserve economic_hypothesis linkage to SignalDefinition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class SignalMetadata:
    """Operational metadata for a registered signal experiment."""

    signal_name: str
    version: str
    source: str = "iqrp.alpha"
    dataset: str = ""
    data_version: str = ""
    pit_compliant: bool = True
    leakage_checks: tuple[str, ...] = ()
    universe: str = "default"
    frequency: str = "1d"
    owner: str = "research"
    economic_hypothesis: str = ""
    tags: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        object.__setattr__(self, "leakage_checks", tuple(self.leakage_checks))
        object.__setattr__(self, "tags", tuple(self.tags))

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_name": self.signal_name,
            "version": self.version,
            "source": self.source,
            "dataset": self.dataset,
            "data_version": self.data_version,
            "pit_compliant": self.pit_compliant,
            "leakage_checks": list(self.leakage_checks),
            "universe": self.universe,
            "frequency": self.frequency,
            "owner": self.owner,
            "economic_hypothesis": self.economic_hypothesis,
            "tags": list(self.tags),
            "extras": dict(self.extras),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignalMetadata:
        def _dt(key: str) -> datetime:
            raw = data.get(key)
            if isinstance(raw, datetime):
                return raw
            if isinstance(raw, str) and raw:
                return datetime.fromisoformat(raw)
            return datetime.now(UTC)

        return cls(
            signal_name=str(data["signal_name"]),
            version=str(data["version"]),
            source=str(data.get("source") or "iqrp.alpha"),
            dataset=str(data.get("dataset") or ""),
            data_version=str(data.get("data_version") or ""),
            pit_compliant=bool(data.get("pit_compliant", True)),
            leakage_checks=tuple(data.get("leakage_checks") or ()),
            universe=str(data.get("universe") or "default"),
            frequency=str(data.get("frequency") or "1d"),
            owner=str(data.get("owner") or "research"),
            economic_hypothesis=str(data.get("economic_hypothesis") or ""),
            tags=tuple(data.get("tags") or ()),
            extras=dict(data.get("extras") or {}),
            created_at=_dt("created_at"),
            updated_at=_dt("updated_at"),
        )
