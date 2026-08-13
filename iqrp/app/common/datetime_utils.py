"""Timezone-aware datetime helpers.

All platform timestamps are UTC unless a caller explicitly converts.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from iqrp.app.core.exceptions import ValidationError


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def to_iso8601(value: datetime | date, *, timespec: str = "seconds") -> str:
    """Serialize a date/datetime to ISO-8601.

    Datetimes are converted to UTC and emitted with a ``Z`` suffix.
    """
    if isinstance(value, datetime):
        utc_value = ensure_utc(value)
        text = utc_value.isoformat(timespec=timespec)
        return text.replace("+00:00", "Z")
    return value.isoformat()


def parse_datetime(value: str | datetime | date) -> datetime:
    """Parse a datetime from ISO-8601 text or pass through datetime/date."""
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if not isinstance(value, str):
        raise ValidationError(
            f"Cannot parse datetime from type {type(value).__name__}",
            code="DATETIME_PARSE_FAILED",
            details={"value": repr(value)},
        )
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValidationError(
            f"Invalid ISO-8601 datetime: {value}",
            code="DATETIME_PARSE_FAILED",
            details={"value": value},
        ) from exc
    return ensure_utc(parsed)


def add_duration(
    value: datetime,
    *,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
) -> datetime:
    """Add a duration to a datetime, returning UTC."""
    base = ensure_utc(value)
    return base + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def to_unix_seconds(value: datetime) -> float:
    """Convert a datetime to Unix epoch seconds (UTC)."""
    return ensure_utc(value).timestamp()


def from_unix_seconds(seconds: float) -> datetime:
    """Convert Unix epoch seconds to a UTC datetime."""
    return datetime.fromtimestamp(seconds, tz=UTC)


def coerce_datetime(value: Any) -> datetime:
    """Best-effort coercion used by validators and loaders."""
    if isinstance(value, (str, datetime, date)):
        return parse_datetime(value)
    if isinstance(value, (int, float)):
        return from_unix_seconds(float(value))
    raise ValidationError(
        f"Unsupported datetime value: {value!r}",
        code="DATETIME_COERCE_FAILED",
        details={"type": type(value).__name__},
    )
