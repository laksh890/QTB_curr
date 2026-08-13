"""Timezone-aware deterministic backtest clock."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterator
from zoneinfo import ZoneInfo


class ClockFrequency(str, Enum):
    """Supported clock advancement granularities."""

    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    CUSTOM = "custom"


_FREQUENCY_ALIASES: dict[str, ClockFrequency] = {
    "tick": ClockFrequency.TICK,
    "second": ClockFrequency.SECOND,
    "seconds": ClockFrequency.SECOND,
    "sec": ClockFrequency.SECOND,
    "s": ClockFrequency.SECOND,
    "minute": ClockFrequency.MINUTE,
    "minutes": ClockFrequency.MINUTE,
    "min": ClockFrequency.MINUTE,
    "m": ClockFrequency.MINUTE,
    "hourly": ClockFrequency.HOURLY,
    "hour": ClockFrequency.HOURLY,
    "hours": ClockFrequency.HOURLY,
    "h": ClockFrequency.HOURLY,
    "daily": ClockFrequency.DAILY,
    "day": ClockFrequency.DAILY,
    "days": ClockFrequency.DAILY,
    "d": ClockFrequency.DAILY,
    "custom": ClockFrequency.CUSTOM,
}


def _parse_frequency(value: ClockFrequency | str) -> ClockFrequency:
    if isinstance(value, ClockFrequency):
        return value
    key = str(value).strip().lower()
    if key not in _FREQUENCY_ALIASES:
        raise ValueError(
            f"unsupported clock frequency {value!r}; "
            f"expected one of {sorted(_FREQUENCY_ALIASES)}"
        )
    return _FREQUENCY_ALIASES[key]


def _resolve_tz(tz: str | ZoneInfo | None) -> ZoneInfo | timezone:
    if tz is None:
        return timezone.utc
    if isinstance(tz, ZoneInfo):
        return tz
    if isinstance(tz, timezone):
        return tz
    name = str(tz).strip()
    if name.upper() in {"UTC", "Z", "GMT"}:
        return timezone.utc
    return ZoneInfo(name)


class BacktestClock:
    """Deterministic, timezone-aware simulation clock.

    Advances in fixed steps (tick / second / minute / hourly / daily / custom).
    Never jumps backwards. All timestamps are timezone-aware.
    """

    __slots__ = (
        "_start",
        "_now",
        "_frequency",
        "_step",
        "_tz",
        "_tick_size",
    )

    def __init__(
        self,
        start: datetime,
        *,
        frequency: ClockFrequency | str = ClockFrequency.DAILY,
        step: timedelta | None = None,
        timezone: str | ZoneInfo | None = "UTC",
        tick_size: timedelta | None = None,
    ) -> None:
        self._tz = _resolve_tz(timezone)
        if start.tzinfo is None:
            start = start.replace(tzinfo=self._tz)
        else:
            start = start.astimezone(self._tz)

        self._frequency = _parse_frequency(frequency)
        self._tick_size = tick_size or timedelta(microseconds=1)

        if self._frequency is ClockFrequency.CUSTOM:
            if step is None:
                raise ValueError("custom frequency requires an explicit step timedelta")
            self._step = step
        elif self._frequency is ClockFrequency.TICK:
            self._step = step or self._tick_size
        elif self._frequency is ClockFrequency.SECOND:
            self._step = step or timedelta(seconds=1)
        elif self._frequency is ClockFrequency.MINUTE:
            self._step = step or timedelta(minutes=1)
        elif self._frequency is ClockFrequency.HOURLY:
            self._step = step or timedelta(hours=1)
        elif self._frequency is ClockFrequency.DAILY:
            self._step = step or timedelta(days=1)
        else:  # pragma: no cover
            raise ValueError(f"unhandled frequency {self._frequency}")

        if self._step <= timedelta(0):
            raise ValueError(f"clock step must be positive, got {self._step}")

        self._start = start
        self._now = start

    @property
    def start(self) -> datetime:
        return self._start

    @property
    def now(self) -> datetime:
        return self._now

    @property
    def current(self) -> datetime:
        """Alias for :attr:`now`."""
        return self._now

    @property
    def frequency(self) -> ClockFrequency:
        return self._frequency

    @property
    def step(self) -> timedelta:
        return self._step

    @property
    def tzinfo(self) -> ZoneInfo | timezone:
        return self._tz

    def reset(self, when: datetime | None = None) -> None:
        """Reset the clock to ``when`` (default: original start)."""
        target = self._start if when is None else when
        if target.tzinfo is None:
            target = target.replace(tzinfo=self._tz)
        else:
            target = target.astimezone(self._tz)
        self._now = target

    def set(self, when: datetime) -> None:
        """Set the clock to ``when`` (must not move backwards)."""
        if when.tzinfo is None:
            when = when.replace(tzinfo=self._tz)
        else:
            when = when.astimezone(self._tz)
        if when < self._now:
            raise ValueError(
                f"BacktestClock cannot move backwards: now={self._now.isoformat()} "
                f"requested={when.isoformat()}"
            )
        self._now = when

    def advance(self, steps: int = 1) -> datetime:
        """Advance the clock by ``steps`` frequency increments. Return new time."""
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps == 0:
            return self._now
        self._now = self._now + self._step * steps
        return self._now

    def advance_to(self, when: datetime) -> datetime:
        """Advance (forward only) to ``when`` without requiring step alignment."""
        self.set(when)
        return self._now

    def tick(self) -> datetime:
        """Advance one step."""
        return self.advance(1)

    def __iter__(self) -> Iterator[datetime]:
        """Infinite iterator yielding successive clock times (does not mutate until next)."""
        while True:
            yield self._now
            self.advance(1)

    def range(self, end: datetime, *, inclusive: bool = True) -> Iterator[datetime]:
        """Yield clock times from current (or start) through ``end``."""
        if end.tzinfo is None:
            end = end.replace(tzinfo=self._tz)
        else:
            end = end.astimezone(self._tz)
        while True:
            if inclusive and self._now > end:
                break
            if not inclusive and self._now >= end:
                break
            yield self._now
            nxt = self._now + self._step
            if inclusive and nxt > end and self._now == end:
                break
            if nxt > end and not inclusive:
                break
            if nxt > end and inclusive and self._now < end:
                # Final partial: stop after yielding current; do not overshoot end
                # for discrete iteration — caller advances explicitly via advance_to.
                self._now = nxt
                if self._now > end:
                    self._now = end
                    break
                continue
            self._now = nxt
            if self._now > end:
                break

    def ensure_aware(self, ts: datetime) -> datetime:
        """Normalize a timestamp into this clock's timezone."""
        if ts.tzinfo is None:
            return ts.replace(tzinfo=self._tz)
        return ts.astimezone(self._tz)


__all__ = ["BacktestClock", "ClockFrequency"]
