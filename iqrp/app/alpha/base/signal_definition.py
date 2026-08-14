"""Signal definition contract for institutional alpha research.

CRITICAL RULES:
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve a signal.
- Every SignalDefinition MUST track an economic_hypothesis explaining *why*
  the relationship should exist (not merely that it appears in sample).
- Signal computation helpers must be point-in-time: use only past windows
  (no future leakage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

SignalDirection = Literal["long_short", "long_only", "short_only", "neutral"]
ExpectedRelationship = Literal["positive", "negative", "nonmonotonic", "unknown"]
SignalType = Literal[
    "momentum",
    "mean_reversion",
    "trend",
    "volatility",
    "volume",
    "cross_sectional",
    "event",
    "alternative",
    "statistical",
    "symbolic",
    "custom",
]


@dataclass(slots=True)
class SignalDefinition:
    """Immutable-by-convention research definition for an alpha candidate.

    ``economic_hypothesis`` is mandatory for promotion beyond RESEARCHING.
    Templates that emit definitions produce research candidates only — they
    do **not** claim profitability.
    """

    name: str
    version: str
    formula: str
    features: tuple[str, ...]
    lookback: int
    horizon: int
    universe: str
    frequency: str
    direction: SignalDirection
    expected_relationship: ExpectedRelationship
    economic_hypothesis: str
    owner: str
    signal_type: SignalType = "custom"
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("SignalDefinition.name must be non-empty")
        if not self.version.strip():
            raise ValueError("SignalDefinition.version must be non-empty")
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        # Empty hypothesis is allowed at CANDIDATE construction; promotion /
        # APPROVED gates (registry.transition / AlphaResearchEngine.approve)
        # still require a substantive economic_hypothesis.
        # Statistical significance alone ≠ alpha.
        # Historical Sharpe alone cannot approve.
        direction = str(self.direction)
        if direction == "long":
            object.__setattr__(self, "direction", "long_only")
        elif direction == "short":
            object.__setattr__(self, "direction", "short_only")
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "tags", tuple(self.tags))

    @property
    def definition_id(self) -> str:
        return f"{self.name}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "definition_id": self.definition_id,
            "formula": self.formula,
            "features": list(self.features),
            "lookback": self.lookback,
            "horizon": self.horizon,
            "universe": self.universe,
            "frequency": self.frequency,
            "direction": self.direction,
            "expected_relationship": self.expected_relationship,
            "economic_hypothesis": self.economic_hypothesis,
            "owner": self.owner,
            "signal_type": self.signal_type,
            "parameters": dict(self.parameters),
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignalDefinition:
        created = data.get("created_at")
        if isinstance(created, str):
            created_at = datetime.fromisoformat(created)
        elif isinstance(created, datetime):
            created_at = created
        else:
            created_at = datetime.now(UTC)
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            formula=str(data.get("formula") or ""),
            features=tuple(data.get("features") or ()),
            lookback=int(data["lookback"]),
            horizon=int(data["horizon"]),
            universe=str(data.get("universe") or "default"),
            frequency=str(data.get("frequency") or "1d"),
            direction=data.get("direction") or "long_short",  # type: ignore[arg-type]
            expected_relationship=data.get("expected_relationship") or "unknown",  # type: ignore[arg-type]
            economic_hypothesis=str(data.get("economic_hypothesis") or ""),
            owner=str(data.get("owner") or "research"),
            signal_type=data.get("signal_type") or "custom",  # type: ignore[arg-type]
            parameters=dict(data.get("parameters") or {}),
            tags=tuple(data.get("tags") or ()),
            created_at=created_at,
            notes=str(data.get("notes") or ""),
        )
