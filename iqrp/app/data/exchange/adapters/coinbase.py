"""Coinbase Exchange REST market-data adapter."""

from __future__ import annotations

from datetime import datetime

from iqrp.app.common.datetime_utils import to_iso8601, utc_now
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
from iqrp.app.data.types import Timeframe, ms_to_utc, timeframe_to_timedelta

_GRANULARITY: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
}


class CoinbaseExchange(BaseExchange):
    """Coinbase Exchange public market-data endpoints."""

    def __init__(self, settings: ExchangeEndpointSettings) -> None:
        super().__init__(settings)

    def normalize_symbol(self, symbol: str) -> str:
        cleaned = symbol.replace("/", "-").upper()
        if "-" not in cleaned:
            # BTCUSDT -> BTC-USDT heuristic for common quotes.
            for quote in ("USDT", "USD", "EUR", "GBP", "BTC", "ETH"):
                if cleaned.endswith(quote) and len(cleaned) > len(quote):
                    return f"{cleaned[: -len(quote)]}-{quote}"
        return cleaned

    def websocket_url(self, symbol: str, channel: str) -> str:
        del symbol, channel
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
        granularity = _GRANULARITY.get(str(timeframe))
        if granularity is None:
            raise DataError(
                f"Unsupported Coinbase timeframe: {timeframe}",
                code="UNSUPPORTED_TIMEFRAME",
            )
        # Coinbase caps candle pages; respect limit via end truncation.
        max_span = timeframe_to_timedelta(timeframe) * min(limit, 300)
        page_end = min(end, start + max_span)
        product = self.normalize_symbol(symbol)
        payload = await self.request_json(
            "GET",
            f"/products/{product}/candles",
            params={
                "granularity": granularity,
                "start": to_iso8601(start),
                "end": to_iso8601(page_end),
            },
        )
        # Coinbase returns [time, low, high, open, close, volume], newest first.
        candles = [
            Candle(
                exchange=self.name,
                symbol=product,
                timeframe=Timeframe(str(timeframe)),
                open_time=ms_to_utc(int(row[0]) * 1000),
                low=float(row[1]),
                high=float(row[2]),
                open=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in reversed(payload)
        ]
        self.log_page("candles", symbol, len(candles))
        return candles

    async def fetch_trades(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[Trade]:
        del start, end
        product = self.normalize_symbol(symbol)
        payload = await self.request_json(
            "GET",
            f"/products/{product}/trades",
            params={"limit": limit},
        )
        trades = [
            Trade(
                exchange=self.name,
                symbol=product,
                trade_id=str(row["trade_id"]),
                timestamp=row["time"],
                price=float(row["price"]),
                size=float(row["size"]),
                side=str(row.get("side", "")).lower() or None,
            )
            for row in payload
        ]
        self.log_page("trades", symbol, len(trades))
        return trades

    async def fetch_orderbook(self, symbol: str, *, depth: int = 20) -> OrderBook:
        product = self.normalize_symbol(symbol)
        level = 2 if depth > 1 else 1
        payload = await self.request_json(
            "GET",
            f"/products/{product}/book",
            params={"level": level},
        )
        bids = payload.get("bids", [])[:depth]
        asks = payload.get("asks", [])[:depth]
        return OrderBook(
            exchange=self.name,
            symbol=product,
            timestamp=utc_now(),
            bids=tuple(OrderBookLevel(price=float(r[0]), size=float(r[1])) for r in bids),
            asks=tuple(OrderBookLevel(price=float(r[0]), size=float(r[1])) for r in asks),
            sequence=int(payload.get("sequence", 0)),
        )

    async def fetch_funding(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[FundingRate]:
        del symbol, start, end, limit
        return []

    async def fetch_open_interest(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[OpenInterest]:
        del symbol, start, end, limit
        return []

    async def fetch_mark_price(self, symbol: str) -> MarkPrice:
        product = self.normalize_symbol(symbol)
        payload = await self.request_json("GET", f"/products/{product}/ticker")
        price = float(payload["price"])
        return MarkPrice(
            exchange=self.name,
            symbol=product,
            timestamp=utc_now(),
            mark_price=price,
            index_price=price,
        )

    async def fetch_index_price(self, symbol: str) -> IndexPrice:
        mark = await self.fetch_mark_price(symbol)
        return IndexPrice(
            exchange=self.name,
            symbol=mark.symbol,
            timestamp=mark.timestamp,
            index_price=mark.mark_price,
        )

    async def fetch_liquidations(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[Liquidation]:
        del symbol, start, end, limit
        return []
