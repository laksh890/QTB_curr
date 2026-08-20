"""In-memory mock exchange for deterministic unit/integration tests."""

from __future__ import annotations

from datetime import datetime

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
from iqrp.app.data.types import Timeframe, timeframe_to_timedelta


class MockExchange(BaseExchange):
    """Synthetic exchange that generates contiguous candles without network I/O."""

    def __init__(
        self,
        settings: ExchangeEndpointSettings | None = None,
        *,
        gap_after: datetime | None = None,
        gap_size: int = 0,
    ) -> None:
        settings = settings or ExchangeEndpointSettings(
            name="mock",
            rest_base_url="https://mock.local",
            ws_base_url="wss://mock.local",
            rate_limit_per_second=1000.0,
        )
        super().__init__(settings)
        self.gap_after = gap_after
        self.gap_size = gap_size
        self.fail_next_requests = 0

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("-", "").upper()

    def websocket_url(self, symbol: str, channel: str) -> str:
        return f"{self.settings.ws_base_url}/{self.normalize_symbol(symbol).lower()}@{channel}"

    async def open(self) -> None:
        self._client = object()  # type: ignore[assignment]

    async def close(self) -> None:
        self._client = None

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
    ) -> object:
        del method, path, params
        return {}

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[Candle]:
        if self.fail_next_requests > 0:
            self.fail_next_requests -= 1
            raise DataError("transient", code="MOCK_TRANSIENT")

        step = timeframe_to_timedelta(timeframe)
        cursor = start
        out: list[Candle] = []
        skipped = 0
        while cursor <= end and len(out) < limit:
            if (
                self.gap_after is not None
                and self.gap_size > 0
                and cursor > self.gap_after
                and skipped < self.gap_size
            ):
                skipped += 1
                cursor = cursor + step
                continue
            price = 100.0 + (len(out) % 50)
            out.append(
                Candle(
                    exchange=self.name,
                    symbol=self.normalize_symbol(symbol),
                    timeframe=Timeframe(str(timeframe)),
                    open_time=cursor,
                    open=price,
                    high=price + 1,
                    low=price - 1,
                    close=price + 0.5,
                    volume=1.0,
                    close_time=cursor + step,
                )
            )
            cursor = cursor + step
        return out

    async def fetch_trades(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[Trade]:
        del end
        return [
            Trade(
                exchange=self.name,
                symbol=self.normalize_symbol(symbol),
                trade_id="1",
                timestamp=start,
                price=100.0,
                size=0.1,
                side="buy",
            )
        ][:limit]

    async def fetch_orderbook(self, symbol: str, *, depth: int = 20) -> OrderBook:
        del depth
        return OrderBook(
            exchange=self.name,
            symbol=self.normalize_symbol(symbol),
            timestamp=utc_now(),
            bids=(OrderBookLevel(price=99.0, size=1.0),),
            asks=(OrderBookLevel(price=101.0, size=1.0),),
            sequence=1,
        )

    async def fetch_funding(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[FundingRate]:
        del end, limit
        return [
            FundingRate(
                exchange=self.name,
                symbol=self.normalize_symbol(symbol),
                timestamp=start,
                funding_rate=0.0001,
            )
        ]

    async def fetch_open_interest(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[OpenInterest]:
        del end, limit
        return [
            OpenInterest(
                exchange=self.name,
                symbol=self.normalize_symbol(symbol),
                timestamp=start,
                open_interest=1000.0,
            )
        ]

    async def fetch_mark_price(self, symbol: str) -> MarkPrice:
        return MarkPrice(
            exchange=self.name,
            symbol=self.normalize_symbol(symbol),
            timestamp=utc_now(),
            mark_price=100.0,
            index_price=100.0,
        )

    async def fetch_index_price(self, symbol: str) -> IndexPrice:
        return IndexPrice(
            exchange=self.name,
            symbol=self.normalize_symbol(symbol),
            timestamp=utc_now(),
            index_price=100.0,
        )

    async def fetch_liquidations(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[Liquidation]:
        del end, limit
        return [
            Liquidation(
                exchange=self.name,
                symbol=self.normalize_symbol(symbol),
                timestamp=start,
                side="sell",
                price=95.0,
                size=1.0,
                order_id="liq-1",
            )
        ]
