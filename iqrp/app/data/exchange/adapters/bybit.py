"""Bybit REST market-data adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from iqrp.app.common.datetime_utils import utc_now
from iqrp.app.config.settings import ExchangeEndpointSettings
from iqrp.app.core.exceptions import DataError
from iqrp.app.data.exchange.base import BaseExchange
from iqrp.app.data.models import (
    Candle,
    FundingRate,
    IndexPrice,
    Liquidation,
    MarkPrice,
    OpenInterest,
    OrderBook,
    OrderBookLevel,
    Trade,
)
from iqrp.app.data.types import Timeframe, ms_to_utc, utc_to_ms

_INTERVAL_MAP: dict[str, str] = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
}


class BybitExchange(BaseExchange):
    """Bybit v5 public market-data endpoints."""

    def __init__(self, settings: ExchangeEndpointSettings) -> None:
        super().__init__(settings)

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("-", "").replace("/", "").upper()

    def websocket_url(self, symbol: str, channel: str) -> str:
        return self.settings.ws_base_url

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[Candle]:
        interval = _INTERVAL_MAP.get(str(timeframe))
        if interval is None:
            raise DataError(
                f"Unsupported Bybit timeframe: {timeframe}",
                code="UNSUPPORTED_TIMEFRAME",
            )
        payload = await self.request_json(
            "GET",
            "/v5/market/kline",
            params={
                "category": "spot",
                "symbol": self.normalize_symbol(symbol),
                "interval": interval,
                "start": utc_to_ms(start),
                "end": utc_to_ms(end),
                "limit": limit,
            },
        )
        rows = payload.get("result", {}).get("list", [])
        # Bybit returns newest-first.
        candles = [self._parse_kline(symbol, timeframe, row) for row in reversed(rows)]
        self.log_page("candles", symbol, len(candles))
        return candles

    def _parse_kline(self, symbol: str, timeframe: Timeframe | str, row: list[Any]) -> Candle:
        open_ms = int(row[0])
        return Candle(
            exchange=self.name,
            symbol=self.normalize_symbol(symbol),
            timeframe=Timeframe(str(timeframe)),
            open_time=ms_to_utc(open_ms),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            quote_volume=float(row[6]) if len(row) > 6 else None,
        )

    async def fetch_trades(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[Trade]:
        del start, end
        payload = await self.request_json(
            "GET",
            "/v5/market/recent-trade",
            params={
                "category": "spot",
                "symbol": self.normalize_symbol(symbol),
                "limit": limit,
            },
        )
        rows = payload.get("result", {}).get("list", [])
        trades = [
            Trade(
                exchange=self.name,
                symbol=self.normalize_symbol(symbol),
                trade_id=str(row["execId"]),
                timestamp=ms_to_utc(int(row["time"])),
                price=float(row["price"]),
                size=float(row["size"]),
                side=str(row.get("side", "")).lower() or None,
            )
            for row in rows
        ]
        self.log_page("trades", symbol, len(trades))
        return trades

    async def fetch_orderbook(self, symbol: str, *, depth: int = 20) -> OrderBook:
        payload = await self.request_json(
            "GET",
            "/v5/market/orderbook",
            params={
                "category": "spot",
                "symbol": self.normalize_symbol(symbol),
                "limit": depth,
            },
        )
        result = payload.get("result", {})
        ts = int(result.get("ts", utc_to_ms(utc_now())))
        return OrderBook(
            exchange=self.name,
            symbol=self.normalize_symbol(symbol),
            timestamp=ms_to_utc(ts),
            bids=tuple(
                OrderBookLevel(price=float(p), size=float(s)) for p, s in result.get("b", [])
            ),
            asks=tuple(
                OrderBookLevel(price=float(p), size=float(s)) for p, s in result.get("a", [])
            ),
            sequence=int(result.get("u", 0)),
        )

    async def fetch_funding(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[FundingRate]:
        payload = await self.request_json(
            "GET",
            "/v5/market/funding/history",
            params={
                "category": "linear",
                "symbol": self.normalize_symbol(symbol),
                "startTime": utc_to_ms(start),
                "endTime": utc_to_ms(end),
                "limit": limit,
            },
        )
        rows = payload.get("result", {}).get("list", [])
        return [
            FundingRate(
                exchange=self.name,
                symbol=self.normalize_symbol(symbol),
                timestamp=ms_to_utc(int(row["fundingRateTimestamp"])),
                funding_rate=float(row["fundingRate"]),
            )
            for row in rows
        ]

    async def fetch_open_interest(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[OpenInterest]:
        payload = await self.request_json(
            "GET",
            "/v5/market/open-interest",
            params={
                "category": "linear",
                "symbol": self.normalize_symbol(symbol),
                "intervalTime": "5min",
                "startTime": utc_to_ms(start),
                "endTime": utc_to_ms(end),
                "limit": limit,
            },
        )
        rows = payload.get("result", {}).get("list", [])
        return [
            OpenInterest(
                exchange=self.name,
                symbol=self.normalize_symbol(symbol),
                timestamp=ms_to_utc(int(row["timestamp"])),
                open_interest=float(row["openInterest"]),
            )
            for row in rows
        ]

    async def fetch_mark_price(self, symbol: str) -> MarkPrice:
        payload = await self.request_json(
            "GET",
            "/v5/market/tickers",
            params={"category": "linear", "symbol": self.normalize_symbol(symbol)},
        )
        rows = payload.get("result", {}).get("list", [])
        if not rows:
            raise DataError("No Bybit ticker", code="EMPTY_TICKER")
        row = rows[0]
        return MarkPrice(
            exchange=self.name,
            symbol=self.normalize_symbol(symbol),
            timestamp=utc_now(),
            mark_price=float(row["markPrice"]),
            index_price=float(row.get("indexPrice", 0.0)),
        )

    async def fetch_index_price(self, symbol: str) -> IndexPrice:
        mark = await self.fetch_mark_price(symbol)
        return IndexPrice(
            exchange=self.name,
            symbol=mark.symbol,
            timestamp=mark.timestamp,
            index_price=float(mark.index_price or 0.0),
        )

    async def fetch_liquidations(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[Liquidation]:
        del start, end, limit, symbol
        # Bybit liquidations are primarily websocket-fed; REST returns empty by design.
        return []
