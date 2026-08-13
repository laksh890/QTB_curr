"""Deterministic priority event queue.

Ordering key: ``(timestamp, priority, sequence)``.

* Earlier timestamps first.
* Within the same timestamp, lower ``priority`` first (MARKET before FILL).
* Insertion sequence breaks remaining ties for full determinism.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

from iqrp.app.backtesting.event_engine.event import Event


@dataclass(order=True, slots=True)
class _QueueItem:
    timestamp: datetime
    priority: int
    sequence: int
    event: Event = field(compare=False)


class EventQueue:
    """Min-heap event queue with deterministic FIFO-within-priority semantics."""

    __slots__ = ("_heap", "_sequence", "_size")

    def __init__(self) -> None:
        self._heap: list[_QueueItem] = []
        self._sequence: int = 0
        self._size: int = 0

    def put(self, event: Event) -> None:
        """Enqueue an event."""
        if not isinstance(event, Event):
            raise TypeError(f"expected Event, got {type(event)!r}")
        item = _QueueItem(
            timestamp=event.timestamp,
            priority=int(event.priority),
            sequence=self._sequence,
            event=event,
        )
        self._sequence += 1
        heapq.heappush(self._heap, item)
        self._size += 1

    def get(self) -> Event:
        """Pop and return the next event. Raises ``IndexError`` if empty."""
        if not self._heap:
            raise IndexError("get from empty EventQueue")
        item = heapq.heappop(self._heap)
        self._size -= 1
        return item.event

    def peek(self) -> Event | None:
        """Return the next event without removing it, or ``None`` if empty."""
        if not self._heap:
            return None
        return self._heap[0].event

    def empty(self) -> bool:
        return self._size == 0

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._heap.clear()
        self._size = 0

    def drain_until(self, asof: datetime) -> list[Event]:
        """Pop all events with ``timestamp <= asof`` in priority order."""
        out: list[Event] = []
        while self._heap and self._heap[0].timestamp <= asof:
            out.append(self.get())
        return out

    def drain_at(self, timestamp: datetime) -> list[Event]:
        """Pop all events exactly at ``timestamp`` in priority order."""
        out: list[Event] = []
        while self._heap and self._heap[0].timestamp == timestamp:
            out.append(self.get())
        return out

    def __iter__(self) -> Iterator[Event]:
        while not self.empty():
            yield self.get()


__all__ = ["EventQueue"]
