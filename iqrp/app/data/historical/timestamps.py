"""Timezone-aware timestamp standardization (UTC storage; no silent naive→UTC)."""

from __future__ import annotations

from typing import Any

import pandas as pd


class NaiveTimestampError(ValueError):
    """Raised when naive timestamps are present without explicit timezone config."""


def ensure_aware_utc(
    series: pd.Series | Any,
    *,
    assume_timezone: str | None = None,
    record: dict[str, Any] | None = None,
) -> pd.Series:
    """Convert timestamps to UTC-aware.

    If values are naive, ``assume_timezone`` **must** be provided; conversion is
    recorded. Never silently treats naive timestamps as UTC.
    """
    ts = pd.to_datetime(series, utc=False)
    # pandas may produce DatetimeIndex or Series
    if isinstance(ts, pd.DatetimeIndex):
        s = pd.Series(ts)
    else:
        s = ts

    # Detect naive: tz is None on dtype or on first valid
    naive = False
    if getattr(s.dt, "tz", None) is None:
        naive = True
    else:
        # mixed — check sample
        sample = s.dropna()
        if not sample.empty and sample.iloc[0].tzinfo is None:
            naive = True

    conversion_note = None
    if naive:
        if not assume_timezone:
            raise NaiveTimestampError(
                "naive timestamps require explicit assume_timezone "
                "(refusing silent UTC assumption)"
            )
        localized = s.dt.tz_localize(assume_timezone)
        out = localized.dt.tz_convert("UTC")
        conversion_note = (
            f"naive timestamps localized as {assume_timezone} then converted to UTC"
        )
    else:
        out = s.dt.tz_convert("UTC")
        conversion_note = "timezone-aware timestamps converted to UTC"

    if record is not None:
        record["timestamp_conversion"] = conversion_note
        record["assume_timezone"] = assume_timezone
        record["stored_timezone"] = "UTC"

    return out


def preserve_exchange_timezone_meta(
    *,
    original_timezone: str,
    exchange_timezone: str,
) -> dict[str, str]:
    return {
        "original_timezone": original_timezone,
        "exchange_timezone": exchange_timezone,
        "stored_timezone": "UTC",
    }


__all__ = [
    "NaiveTimestampError",
    "ensure_aware_utc",
    "preserve_exchange_timezone_meta",
]
