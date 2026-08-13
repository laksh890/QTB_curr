"""Deterministic event-driven backtest engine.

CRITICAL — Point-in-time correctness
------------------------------------
Before every handler invocation the engine advances the clock to
``event.timestamp``. Handlers **must not** read any data with an effective
timestamp strictly after that clock value. Use
:func:`iqrp.app.backtesting.pit.assert_no_lookahead` (and related helpers)
at every data access boundary. Violations should invalidate the backtest.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Callable

from iqrp.app.backtesting.event_engine.clock import BacktestClock
from iqrp.app.backtesting.event_engine.event import Event, EventType
from iqrp.app.backtesting.event_engine.event_queue import EventQueue
from iqrp.app.backtesting.event_engine.scheduler import EventScheduler
from iqrp.app.backtesting.types import BacktestState

EventHandler = Callable[[Event], None]


class LookaheadError(RuntimeError):
    """Raised when an event would be processed after the simulation end or
    when the engine detects an out-of-order / future-data violation."""


class EventDrivenEngine:
    """Priority-queue event loop for institutional backtests.

    Parameters
    ----------
    clock:
        Deterministic timezone-aware :class:`BacktestClock`.
    queue:
        Optional pre-built queue; a fresh :class:`EventQueue` is created otherwise.
    scheduler:
        Optional :class:`EventScheduler` for recurring events.
    """

    __slots__ = (
        "clock",
        "queue",
        "scheduler",
        "_handlers",
        "_wildcard_handlers",
        "_state",
        "_processed",
        "_on_invalidate",
    )

    def __init__(
        self,
        *,
        clock: BacktestClock,
        queue: EventQueue | None = None,
        scheduler: EventScheduler | None = None,
        on_invalidate: Callable[[str], None] | None = None,
    ) -> None:
        self.clock = clock
        self.queue = queue if queue is not None else EventQueue()
        self.scheduler = scheduler if scheduler is not None else EventScheduler()
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._wildcard_handlers: list[EventHandler] = []
        self._state = BacktestState.CREATED
        self._processed = 0
        self._on_invalidate = on_invalidate

    @property
    def state(self) -> BacktestState:
        return self._state

    @property
    def processed_count(self) -> int:
        return self._processed

    def register(
        self,
        event_type: EventType | str | None,
        handler: EventHandler,
    ) -> None:
        """Register a handler for ``event_type``.

        Pass ``event_type=None`` to receive every event (after type-specific
        handlers).
        """
        if event_type is None:
            self._wildcard_handlers.append(handler)
            return
        et = event_type if isinstance(event_type, EventType) else EventType(str(event_type))
        self._handlers[et].append(handler)

    def unregister(
        self,
        event_type: EventType | str | None,
        handler: EventHandler,
    ) -> bool:
        """Remove a previously registered handler. Returns whether it was found."""
        if event_type is None:
            try:
                self._wildcard_handlers.remove(handler)
                return True
            except ValueError:
                return False
        et = event_type if isinstance(event_type, EventType) else EventType(str(event_type))
        handlers = self._handlers.get(et, [])
        try:
            handlers.remove(handler)
            return True
        except ValueError:
            return False

    def submit(self, event: Event) -> None:
        """Enqueue a single event."""
        self.queue.put(event)

    def invalidate(self, reason: str) -> None:
        """Mark the backtest as invalidated (e.g. look-ahead detected)."""
        self._state = BacktestState.INVALIDATED
        if self._on_invalidate is not None:
            self._on_invalidate(reason)

    def _dispatch(self, event: Event) -> None:
        # Advance clock to the event time before any handler runs so that
        # clock.now is a reliable PIT boundary.
        if event.timestamp < self.clock.now:
            # Allow equal-time batching; reject true time-travel.
            # Events can share a timestamp; clock may already be at that time.
            pass
        if event.timestamp > self.clock.now:
            self.clock.advance_to(event.timestamp)
        elif event.timestamp != self.clock.now:
            # event.timestamp < clock.now → out-of-order / late event
            raise LookaheadError(
                f"out-of-order event {event.event_id}: "
                f"event.ts={event.timestamp.isoformat()} "
                f"clock.now={self.clock.now.isoformat()}"
            )

        for handler in self._handlers.get(event.event_type, ()):
            handler(event)
        for handler in self._wildcard_handlers:
            handler(event)
        self._processed += 1

    def run(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        *,
        advance_empty_ticks: bool = False,
        max_events: int | None = None,
    ) -> BacktestState:
        """Run the event loop from ``start`` through ``end``.

        Algorithm
        ---------
        1. Optionally reset/advance the clock to ``start``.
        2. Seed due scheduled jobs.
        3. While the next queued event has ``timestamp <= end``:
           a. Advance clock to that timestamp.
           b. Enqueue any scheduler jobs due at/before that time.
           c. Drain all events at that exact timestamp in priority order.
        4. Optionally advance empty ticks when ``advance_empty_ticks`` is set.
        5. Advance clock to ``end`` and mark ``COMPLETED`` (unless invalidated).

        Parameters
        ----------
        start:
            Simulation start. Defaults to the clock's current time.
        end:
            Inclusive simulation end. Required unless the queue is empty and
            no further work is expected — callers should always pass ``end``.
        advance_empty_ticks:
            If True, step the clock by its frequency even when the queue is
            idle (useful for time-based scheduled jobs).
        max_events:
            Optional safety cap on processed events.
        """
        if end is None:
            raise ValueError("EventDrivenEngine.run requires an end timestamp")

        end = self.clock.ensure_aware(end)
        if start is not None:
            start = self.clock.ensure_aware(start)
            if start < self.clock.now:
                self.clock.reset(start)
            elif start > self.clock.now:
                self.clock.advance_to(start)

        if self._state is BacktestState.INVALIDATED:
            return self._state

        self._state = BacktestState.RUNNING

        try:
            # Materialize recurring jobs up front for determinism.
            self.scheduler.seed_until(
                self.queue, start=self.clock.now, end=end, clock=self.clock
            )

            while self._state is BacktestState.RUNNING:
                if max_events is not None and self._processed >= max_events:
                    break

                nxt = self.queue.peek()
                if nxt is None:
                    if not advance_empty_ticks:
                        break
                    # Idle tick advancement for time-driven schedules.
                    nxt_time = self.clock.now + self.clock.step
                    if nxt_time > end:
                        break
                    self.clock.advance_to(nxt_time)
                    self.scheduler.enqueue_due(self.queue, self.clock.now)
                    continue

                if nxt.timestamp > end:
                    break

                # Past / out-of-order events must not spin the loop forever.
                if nxt.timestamp < self.clock.now:
                    raise LookaheadError(
                        f"out-of-order event {nxt.event_id}: "
                        f"event.ts={nxt.timestamp.isoformat()} "
                        f"clock.now={self.clock.now.isoformat()}"
                    )

                # Jump clock to next event cluster.
                if nxt.timestamp > self.clock.now:
                    self.clock.advance_to(nxt.timestamp)

                self.scheduler.enqueue_due(self.queue, self.clock.now)

                batch = self.queue.drain_at(self.clock.now)
                if not batch:
                    # Due to newly enqueued jobs at a different micro-ordering,
                    # fall through to peek again.
                    continue
                for event in batch:
                    if self._state is BacktestState.INVALIDATED:
                        break
                    self._dispatch(event)

            if self._state is BacktestState.RUNNING:
                if self.clock.now < end:
                    self.clock.advance_to(end)
                self._state = BacktestState.COMPLETED
        except Exception:
            self._state = BacktestState.FAILED
            raise

        return self._state


__all__ = ["EventDrivenEngine", "EventHandler", "LookaheadError"]
