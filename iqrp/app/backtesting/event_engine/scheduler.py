"""Recurring event scheduler for the backtest event engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from iqrp.app.backtesting.event_engine.clock import BacktestClock
from iqrp.app.backtesting.event_engine.event import Event, EventType
from iqrp.app.backtesting.event_engine.event_queue import EventQueue

EventFactory = Callable[[datetime], Event]


@dataclass(slots=True)
class ScheduledJob:
    """A recurring job that emits events on a fixed interval."""

    job_id: str
    interval: timedelta
    factory: EventFactory
    next_time: datetime
    end: datetime | None = None
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EventScheduler:
    """Schedule recurring events onto an :class:`EventQueue`.

    Jobs are deterministic: given the same start/interval and insertion order,
    emitted event timestamps are identical across runs.
    """

    __slots__ = ("_jobs",)

    def __init__(self) -> None:
        self._jobs: dict[str, ScheduledJob] = {}

    def schedule(
        self,
        *,
        interval: timedelta,
        factory: EventFactory,
        start: datetime,
        end: datetime | None = None,
        job_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Register a recurring job. Returns the job id."""
        if interval <= timedelta(0):
            raise ValueError(f"interval must be positive, got {interval}")
        if start.tzinfo is None:
            raise ValueError("schedule start must be timezone-aware")
        if end is not None and end.tzinfo is None:
            raise ValueError("schedule end must be timezone-aware")
        jid = job_id or uuid4().hex
        self._jobs[jid] = ScheduledJob(
            job_id=jid,
            interval=interval,
            factory=factory,
            next_time=start,
            end=end,
            metadata=dict(metadata or {}),
        )
        return jid

    def schedule_event_type(
        self,
        event_type: EventType | str,
        *,
        interval: timedelta,
        start: datetime,
        end: datetime | None = None,
        payload: Mapping[str, Any] | None = None,
        job_id: str | None = None,
    ) -> str:
        """Convenience: schedule plain :class:`Event` instances of a given type."""
        et = event_type if isinstance(event_type, EventType) else EventType(str(event_type))
        base_payload = dict(payload or {})

        def _factory(ts: datetime) -> Event:
            return Event(timestamp=ts, event_type=et, payload=dict(base_payload))

        return self.schedule(
            interval=interval,
            factory=_factory,
            start=start,
            end=end,
            job_id=job_id,
        )

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.enabled = False
        return True

    def remove(self, job_id: str) -> bool:
        return self._jobs.pop(job_id, None) is not None

    def jobs(self) -> list[ScheduledJob]:
        return list(self._jobs.values())

    def enqueue_due(self, queue: EventQueue, asof: datetime) -> list[Event]:
        """Enqueue all job occurrences with ``next_time <= asof``.

        Advances each job's ``next_time`` past ``asof``.
        """
        emitted: list[Event] = []
        for job in self._jobs.values():
            if not job.enabled:
                continue
            while job.next_time <= asof:
                if job.end is not None and job.next_time > job.end:
                    job.enabled = False
                    break
                event = job.factory(job.next_time)
                queue.put(event)
                emitted.append(event)
                job.next_time = job.next_time + job.interval
                if job.end is not None and job.next_time > job.end:
                    job.enabled = False
                    break
        return emitted

    def seed_until(
        self,
        queue: EventQueue,
        *,
        start: datetime,
        end: datetime,
        clock: BacktestClock | None = None,
    ) -> int:
        """Materialize all scheduled occurrences in ``[start, end]`` onto ``queue``.

        If ``clock`` is provided, times are normalized into the clock timezone.
        """
        count = 0
        for job in self._jobs.values():
            if not job.enabled:
                continue
            t = job.next_time
            if clock is not None:
                t = clock.ensure_aware(t)
                start_n = clock.ensure_aware(start)
                end_n = clock.ensure_aware(end)
            else:
                start_n, end_n = start, end
            if t < start_n:
                # Fast-forward to first occurrence >= start
                if job.interval <= timedelta(0):
                    raise ValueError("invalid interval")
                delta = start_n - t
                steps = delta // job.interval
                t = t + job.interval * steps
                if t < start_n:
                    t = t + job.interval
            while t <= end_n:
                job_end = job.end
                if job_end is not None and t > job_end:
                    break
                queue.put(job.factory(t))
                count += 1
                t = t + job.interval
            job.next_time = t
            if job.end is not None and job.next_time > job.end:
                job.enabled = False
        return count


__all__ = ["EventFactory", "EventScheduler", "ScheduledJob"]
