"""Shared market-data enums and timeframe helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum


class ExchangeId(StrEnum):
    """Canonical exchange identifiers (factory keys)."""

    BINANCE = "binance"
    BYBIT = "bybit"
    COINBASE = "coinbase"


class MarketDataType(StrEnum):
    """Supported market-data product types."""

    CANDLE = "candles"
    TRADE = "trades"
    ORDERBOOK = "orderbook"
    FUNDING = "funding"
    OPEN_INTEREST = "open_interest"
    MARK_PRICE = "mark_price"
    INDEX_PRICE = "index_price"
    LIQUIDATION = "liquidations"


class Timeframe(StrEnum):
    """Canonical OHLCV timeframes."""

    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H12 = "12h"
    D1 = "1d"
    W1 = "1w"


_TIMEFRAME_DELTA: dict[Timeframe, timedelta] = {
    Timeframe.M1: timedelta(minutes=1),
    Timeframe.M3: timedelta(minutes=3),
    Timeframe.M5: timedelta(minutes=5),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.M30: timedelta(minutes=30),
    Timeframe.H1: timedelta(hours=1),
    Timeframe.H2: timedelta(hours=2),
    Timeframe.H4: timedelta(hours=4),
    Timeframe.H6: timedelta(hours=6),
    Timeframe.H12: timedelta(hours=12),
    Timeframe.D1: timedelta(days=1),
    Timeframe.W1: timedelta(weeks=1),
}


def timeframe_to_timedelta(timeframe: Timeframe | str) -> timedelta:
    """Return the duration of a timeframe."""
    tf = Timeframe(str(timeframe))
    return _TIMEFRAME_DELTA[tf]


def timeframe_to_ms(timeframe: Timeframe | str) -> int:
    """Return timeframe duration in milliseconds."""
    return int(timeframe_to_timedelta(timeframe).total_seconds() * 1000)


def ms_to_utc(ms: int | float) -> datetime:
    """Convert Unix epoch milliseconds to timezone-aware UTC."""
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=UTC)


def utc_to_ms(value: datetime) -> int:
    """Convert a timezone-aware datetime to Unix epoch milliseconds."""
    value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return int(value.timestamp() * 1000)
