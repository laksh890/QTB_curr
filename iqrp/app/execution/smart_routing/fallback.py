"""Venue fallback chain when primary routing fails.

Builds an ordered list of alternate venues and selects the next
eligible venue after a failure (reject, timeout, halt, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from iqrp.app.execution.smart_routing.scoring import VenueScore
from iqrp.app.execution.smart_routing.venue import Venue


@dataclass(slots=True)
class FallbackStep:
    """One step in the fallback chain."""

    venue_id: str
    score: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "score": float(self.score),
            "reason": self.reason,
        }


@dataclass(slots=True)
class FallbackChain:
    """Ordered fallback venues after the primary."""

    primary_venue_id: str
    steps: list[FallbackStep] = field(default_factory=list)
    cursor: int = 0

    def next_venue(self) -> FallbackStep | None:
        """Advance and return the next fallback venue, or None if exhausted."""
        if self.cursor >= len(self.steps):
            return None
        step = self.steps[self.cursor]
        self.cursor += 1
        return step

    def peek(self) -> FallbackStep | None:
        if self.cursor >= len(self.steps):
            return None
        return self.steps[self.cursor]

    def remaining(self) -> list[FallbackStep]:
        return list(self.steps[self.cursor :])

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_venue_id": self.primary_venue_id,
            "steps": [s.to_dict() for s in self.steps],
            "cursor": self.cursor,
        }


def build_fallback_chain(
    scores: list[VenueScore],
    *,
    primary_venue_id: str | None = None,
    exclude: Iterable[str] | None = None,
    max_fallbacks: int = 5,
) -> FallbackChain:
    """Build a fallback chain from ranked venue scores.

    Primary is the top-scoring venue (or ``primary_venue_id`` if provided).
    Remaining venues form the fallback steps in score order.
    """
    ranked = sorted(scores, key=lambda s: s.score, reverse=True)
    excluded = {str(x) for x in (exclude or [])}

    if not ranked:
        return FallbackChain(primary_venue_id=primary_venue_id or "")

    if primary_venue_id is None:
        primary_venue_id = ranked[0].venue_id

    steps: list[FallbackStep] = []
    for s in ranked:
        if s.venue_id == primary_venue_id:
            continue
        if s.venue_id in excluded:
            continue
        steps.append(
            FallbackStep(
                venue_id=s.venue_id,
                score=s.score,
                reason="ranked_fallback",
            )
        )
        if len(steps) >= int(max_fallbacks):
            break

    return FallbackChain(primary_venue_id=str(primary_venue_id), steps=steps)


def select_fallback(
    chain: FallbackChain,
    venues: dict[str, Venue],
    *,
    failed_venue_id: str,
    failure_reason: str = "primary_failed",
) -> Venue | None:
    """Select the next routable venue from the chain after a failure.

    Skips venues that are no longer routable. Mutates chain cursor.
    """
    del failed_venue_id  # caller responsibility to mark failed; chain already excludes used
    while True:
        step = chain.next_venue()
        if step is None:
            return None
        venue = venues.get(step.venue_id)
        if venue is None:
            continue
        if not venue.venue_state.is_routable():
            continue
        step.reason = failure_reason
        return venue


__all__ = [
    "FallbackStep",
    "FallbackChain",
    "build_fallback_chain",
    "select_fallback",
]
