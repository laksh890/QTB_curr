"""Configurable exchange calendars / trading sessions (no invented sessions)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd


@dataclass(frozen=True)
class SessionSpec:
    """One trading session definition in exchange-local time."""

    start: time
    end: time  # exclusive end for bar generation (e.g. 15:30 → last 1m bar 15:29)
    label: str = "regular"


@dataclass
class ExchangeCalendar:
    """Exchange calendar with weekends, holidays, and special sessions.

    Holidays / special sessions must be configured explicitly — never invented.
    Set ``continuous_24x7=True`` for crypto-style markets (UTC, every day, full day).
    """

    exchange_id: str
    timezone: str
    regular_session: SessionSpec
    weekly_closed: frozenset[int] = frozenset({5, 6})  # Sat=5 Sun=6
    holidays: frozenset[date] = frozenset()
    special_sessions: dict[date, SessionSpec] = field(default_factory=dict)
    early_closes: dict[date, time] = field(default_factory=dict)
    late_opens: dict[date, time] = field(default_factory=dict)
    continuous_24x7: bool = False
    market_type: str = "EQUITY"

    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def is_weekend(self, d: date) -> bool:
        if self.continuous_24x7:
            return False
        return int(d.weekday()) in self.weekly_closed

    def is_holiday(self, d: date) -> bool:
        return d in self.holidays

    def is_trading_day(self, d: date) -> bool:
        if self.continuous_24x7:
            return d not in self.holidays
        if self.is_weekend(d) or self.is_holiday(d):
            return False
        return True

    def session_for(self, d: date) -> SessionSpec | None:
        if not self.is_trading_day(d) and d not in self.special_sessions:
            return None
        if d in self.special_sessions:
            return self.special_sessions[d]
        if self.continuous_24x7:
            return SessionSpec(start=time(0, 0), end=time(0, 0), label="24x7")
        start = self.late_opens.get(d, self.regular_session.start)
        end = self.early_closes.get(d, self.regular_session.end)
        return SessionSpec(start=start, end=end, label=self.regular_session.label)

    def expected_bar_timestamps(
        self,
        d: date,
        frequency_seconds: int,
    ) -> list[pd.Timestamp]:
        """Expected bar open timestamps for a session (exchange tz)."""
        sess = self.session_for(d)
        if sess is None:
            return []
        tz = self.tz()
        start_dt = datetime.combine(d, sess.start, tzinfo=tz)
        if self.continuous_24x7 or (
            sess.label == "24x7" and sess.start == time(0, 0) and sess.end == time(0, 0)
        ):
            end_dt = start_dt + timedelta(days=1)
        else:
            end_dt = datetime.combine(d, sess.end, tzinfo=tz)
        if end_dt <= start_dt:
            return []
        step = timedelta(seconds=int(frequency_seconds))
        out: list[pd.Timestamp] = []
        cur = start_dt
        while cur < end_dt:
            out.append(pd.Timestamp(cur))
            cur += step
        return out

    def trading_days(self, start: date, end: date) -> list[date]:
        days: list[date] = []
        cur = start
        while cur <= end:
            if self.is_trading_day(cur) or cur in self.special_sessions:
                if self.session_for(cur) is not None:
                    days.append(cur)
            cur += timedelta(days=1)
        return days

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange_id": self.exchange_id,
            "timezone": self.timezone,
            "market_type": self.market_type,
            "continuous_24x7": self.continuous_24x7,
            "regular_session": {
                "start": self.regular_session.start.isoformat(),
                "end": self.regular_session.end.isoformat(),
                "label": self.regular_session.label,
            },
            "weekly_closed": sorted(self.weekly_closed),
            "holidays": [d.isoformat() for d in sorted(self.holidays)],
            "special_sessions": {
                d.isoformat(): {
                    "start": s.start.isoformat(),
                    "end": s.end.isoformat(),
                    "label": s.label,
                }
                for d, s in self.special_sessions.items()
            },
            "early_closes": {d.isoformat(): t.isoformat() for d, t in self.early_closes.items()},
            "late_opens": {d.isoformat(): t.isoformat() for d, t in self.late_opens.items()},
            "note": (
                "Holidays/special sessions must be configured; "
                "missing sessions are not invented. "
                "CRYPTO continuous markets use UTC 24x7 — do not apply equity sessions."
            ),
        }


# Known NSE cash equity regular session (local IST). Holiday list is partial /
# configurable — unknown holidays are NOT invented as trading days when listed.
_NSE_KNOWN_HOLIDAYS_2026: frozenset[date] = frozenset(
    {
        # Leave mostly empty: do not invent. Callers may extend via config.
    }
)


def nse_equity_calendar(
    *,
    holidays: Iterable[date] | None = None,
    early_closes: dict[date, time] | None = None,
    late_opens: dict[date, time] | None = None,
    special_sessions: dict[date, SessionSpec] | None = None,
) -> ExchangeCalendar:
    """NSE cash equity / index regular session calendar (configurable holidays)."""
    hol = frozenset(holidays) if holidays is not None else _NSE_KNOWN_HOLIDAYS_2026
    return ExchangeCalendar(
        exchange_id="NSE",
        timezone="Asia/Kolkata",
        regular_session=SessionSpec(start=time(9, 15), end=time(15, 30), label="regular"),
        holidays=hol,
        early_closes=dict(early_closes or {}),
        late_opens=dict(late_opens or {}),
        special_sessions=dict(special_sessions or {}),
        continuous_24x7=False,
        market_type="EQUITY",
    )


def crypto_24x7_calendar() -> ExchangeCalendar:
    """Continuous crypto market calendar (UTC, every calendar day, 00:00–24:00).

    Do NOT apply equity/NSE session rules to crypto.
    """
    return ExchangeCalendar(
        exchange_id="CRYPTO_24x7",
        timezone="UTC",
        regular_session=SessionSpec(start=time(0, 0), end=time(0, 0), label="24x7"),
        weekly_closed=frozenset(),
        holidays=frozenset(),
        continuous_24x7=True,
        market_type="CRYPTO",
    )


FREQUENCY_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "60m": 3600,
    "1D": 86400,
    "1d": 86400,
}


def frequency_to_seconds(freq: str) -> int:
    f = str(freq).strip()
    if f in FREQUENCY_SECONDS:
        return FREQUENCY_SECONDS[f]
    # pandas-like
    if f.endswith("min"):
        return int(f.replace("min", "")) * 60
    raise ValueError(f"unsupported frequency: {freq!r}")


# Market open/close handling assumptions (documented)
SESSION_BOUNDARY_ASSUMPTIONS = """
NSE regular cash session (configurable): 09:15–15:30 Asia/Kolkata.
- First bar: opens at session start (e.g. 09:15 for 1m).
- Last bar: opens at session_end - frequency (e.g. 15:29 for 1m).
- Opening/closing auction effects are NOT modelled; we store provider bars as-is.
- Bars are NEVER manufactured to fill session boundaries.
- Resampling does NOT cross session / overnight boundaries for equity calendars.

CRYPTO 24x7 (UTC):
- market_type=CRYPTO, continuous_market=true, session_model=24x7
- Every UTC calendar day is a trading day (00:00 → next 00:00).
- Do NOT apply NIFTY/NSE session calendars to crypto.
- Resampling uses the same OHLCV aggregation, bucketed by UTC day / continuous UTC time.
"""


__all__ = [
    "FREQUENCY_SECONDS",
    "SESSION_BOUNDARY_ASSUMPTIONS",
    "ExchangeCalendar",
    "SessionSpec",
    "crypto_24x7_calendar",
    "frequency_to_seconds",
    "nse_equity_calendar",
]
