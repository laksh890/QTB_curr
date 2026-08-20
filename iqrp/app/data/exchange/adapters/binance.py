"""Binance REST market-data adapter."""

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
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1d",
    "1w": "1w",
}


class BinanceExchange(BaseExchange):
    """Binance Spot / USD-M public market-data endpoints."""

    def __init__(self, settings: ExchangeEndpointSettings) -> None:
        super().__init__(settings)

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("-", "").replace("/", "").upper()

    def websocket_url(self, symbol: str, channel: str) -> str:
        stream = f"{self.normalize_symbol(symbol).lower()}@{channel}"
        return f"{self.settings.ws_base_url.rstrip('/')}/{stream}"

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
                f"Unsupported Binance timeframe: {timeframe}",
                code="UNSUPPORTED_TIMEFRAME",
            )
        payload = await self.request_json(
            "GET",
            "/api/v3/klines",
            params={
                "symbol": self.normalize_symbol(symbol),
                "interval": interval,
                "startTime": utc_to_ms(start),
                "endTime": utc_to_ms(end),
                "limit": limit,
            },
        )
        candles = [self._parse_kline(symbol, timeframe, row) for row in payload]
        self.log_page("candles", symbol, len(candles))
        return candles

    def _parse_kline(self, symbol: str, timeframe: Timeframe | str, row: list[Any]) -> Candle:
        return Candle(
            exchange=self.name,
            symbol=self.normalize_symbol(symbol),
            timeframe=Timeframe(str(timeframe)),
            open_time=ms_to_utc(int(row[0])),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            close_time=ms_to_utc(int(row[6])),
            quote_volume=float(row[7]),
            trade_count=int(row[8]),
        )

    async def fetch_trades(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[Trade]:
        payload = await self.request_json(
            "GET",
            "/api/v3/aggTrades",
            params={
                "symbol": self.normalize_symbol(symbol),
                "startTime": utc_to_ms(start),
                "endTime": utc_to_ms(end),
                "limit": limit,
            },
        )
        trades = [
            Trade(
                exchange=self.name,
                symbol=self.normalize_symbol(symbol),
                trade_id=str(row["a"]),
                timestamp=ms_to_utc(int(row["T"])),
                price=float(row["p"]),
                size=float(row["q"]),
                is_buyer_maker=bool(row["m"]),
                side="sell" if row["m"] else "buy",
            )
            for row in payload
        ]
        self.log_page("trades", symbol, len(trades))
        return trades

    async def fetch_orderbook(self, symbol: str, *, depth: int = 20) -> OrderBook:
        payload = await self.request_json(
            "GET",
            "/api/v3/depth",
            params={"symbol": self.normalize_symbol(symbol), "limit": depth},
        )
        event_ms = int(payload["E"]) if "E" in payload else utc_to_ms(utc_now())
        return OrderBook(
            exchange=self.name,
            symbol=self.normalize_symbol(symbol),
            timestamp=ms_to_utc(event_ms),
            bids=tuple(OrderBookLevel(price=float(p), size=float(s)) for p, s in payload["bids"]),
            asks=tuple(OrderBookLevel(price=float(p), size=float(s)) for p, s in payload["asks"]),
            sequence=int(payload.get("lastUpdateId", 0)),
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
            "/fapi/v1/fundingRate",
            params={
                "symbol": self.normalize_symbol(symbol),
                "startTime": utc_to_ms(start),
                "endTime": utc_to_ms(end),
                "limit": limit,
            },
        )
        return [
            FundingRate(
                exchange=self.name,
                symbol=self.normalize_symbol(symbol),
                timestamp=ms_to_utc(int(row["fundingTime"])),
                funding_rate=float(row["fundingRate"]),
                mark_price=float(row["markPrice"]) if "markPrice" in row else None,
            )
            for row in payload
        ]

    async def fetch_open_interest(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[OpenInterest]:
        del limit
        payload = await self.request_json(
            "GET",
            "/futures/data/openInterestHist",
            params={
                "symbol": self.normalize_symbol(symbol),
                "period": "5m",
                "startTime": utc_to_ms(start),
                "endTime": utc_to_ms(end),
            },
        )
        return [
            OpenInterest(
                exchange=self.name,
                symbol=self.normalize_symbol(symbol),
                timestamp=ms_to_utc(int(row["timestamp"])),
                open_interest=float(row["sumOpenInterest"]),
                open_interest_value=float(row.get("sumOpenInterestValue", 0.0)),
            )
            for row in payload
        ]

    async def fetch_mark_price(self, symbol: str) -> MarkPrice:
        payload = await self.request_json(
            "GET",
            "/fapi/v1/premiumIndex",
            params={"symbol": self.normalize_symbol(symbol)},
        )
        return MarkPrice(
            exchange=self.name,
            symbol=self.normalize_symbol(symbol),
            timestamp=ms_to_utc(int(payload["time"])),
            mark_price=float(payload["markPrice"]),
            index_price=float(payload["indexPrice"]),
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
        del start, end
        payload = await self.request_json(
            "GET",
            "/fapi/v1/allForceOrders",
            params={"symbol": self.normalize_symbol(symbol), "limit": limit},
        )
        return [
            Liquidation(
                exchange=self.name,
                symbol=self.normalize_symbol(symbol),
                timestamp=ms_to_utc(int(row["time"])),
                side=str(row["side"]).lower(),
                price=float(row["price"]),
                size=float(row["origQty"]),
                order_id=str(row.get("orderId")),
            )
            for row in payload
        ]
