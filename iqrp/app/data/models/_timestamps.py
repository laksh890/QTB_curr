"""Timestamp normalization helpers for market-data models."""

from __future__ import annotations

from datetime import UTC, datetime

from iqrp.app.core.exceptions import ValidationError


def ensure_utc_ms(value: datetime | int | float | str) -> datetime:
    """Normalize timestamps to timezone-aware UTC (millisecond precision input OK)."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        seconds = float(value) / 1000.0 if float(value) >= 1_000_000_000_000 else float(value)
        return datetime.fromtimestamp(seconds, tz=UTC)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raise ValidationError(
        f"Unsupported timestamp type: {type(value).__name__}",
        code="BAD_TIMESTAMP",
    )
