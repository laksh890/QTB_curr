"""Abstract exchange adapter contract."""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Any

import httpx
from loguru import logger

from iqrp.app.config.settings import ExchangeEndpointSettings
from iqrp.app.core.exceptions import DataError
from iqrp.app.data.models import (
    Candle,
    FundingRate,
    IndexPrice,
    Liquidation,
    MarkPrice,
    OpenInterest,
    OrderBook,
    Trade,
)
from iqrp.app.data.rate_limiter import AsyncRateLimiter
from iqrp.app.data.types import Timeframe


class BaseExchange(abc.ABC):
    """Exchange-agnostic market-data adapter.

    Concrete adapters implement endpoint paths and response parsing only.
    HTTP transport, rate limiting, and error mapping live here.
    """

    name: str

    def __init__(self, settings: ExchangeEndpointSettings) -> None:
        self.settings = settings
        self.name = settings.name
        self._limiter = AsyncRateLimiter(settings.rate_limit_per_second)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> BaseExchange:
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def open(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.rest_base_url.rstrip("/"),
                timeout=self.settings.request_timeout_seconds,
                headers={"User-Agent": "iqrp-data/0.1"},
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise DataError(
                f"Exchange '{self.name}' client is not open",
                code="EXCHANGE_NOT_OPEN",
                details={"exchange": self.name},
            )
        return self._client

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Issue a rate-limited HTTP request and return parsed JSON."""
        await self._limiter.acquire()
        try:
            response = await self.client.request(method, path, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise DataError(
                f"Exchange HTTP error: {exc.response.status_code}",
                code="EXCHANGE_HTTP_ERROR",
                details={
                    "exchange": self.name,
                    "path": path,
                    "status": exc.response.status_code,
                    "body": exc.response.text[:500],
                },
            ) from exc
        except httpx.HTTPError as exc:
            raise DataError(
                f"Exchange transport error: {exc}",
                code="EXCHANGE_TRANSPORT_ERROR",
                details={"exchange": self.name, "path": path},
            ) from exc

    @abc.abstractmethod
    async def fetch_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[Candle]:
        """Fetch one page of OHLCV candles in ``[start, end]``."""

    @abc.abstractmethod
    async def fetch_trades(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[Trade]:
        """Fetch one page of trades."""

    @abc.abstractmethod
    async def fetch_orderbook(self, symbol: str, *, depth: int = 20) -> OrderBook:
        """Fetch a top-of-book snapshot."""

    @abc.abstractmethod
    async def fetch_funding(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[FundingRate]:
        """Fetch funding rate history."""

    @abc.abstractmethod
    async def fetch_open_interest(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[OpenInterest]:
        """Fetch open interest history."""

    @abc.abstractmethod
    async def fetch_mark_price(self, symbol: str) -> MarkPrice:
        """Fetch the latest mark price."""

    @abc.abstractmethod
    async def fetch_index_price(self, symbol: str) -> IndexPrice:
        """Fetch the latest index price."""

    @abc.abstractmethod
    async def fetch_liquidations(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> list[Liquidation]:
        """Fetch liquidation events."""

    @abc.abstractmethod
    def normalize_symbol(self, symbol: str) -> str:
        """Map a canonical symbol (e.g. BTCUSDT) to the exchange-native form."""

    @abc.abstractmethod
    def websocket_url(self, symbol: str, channel: str) -> str:
        """Build a websocket subscription URL for ``channel``."""

    def log_page(self, product: str, symbol: str, count: int) -> None:
        logger.debug(
            "exchange_page exchange={} product={} symbol={} count={}",
            self.name,
            product,
            symbol,
            count,
        )
