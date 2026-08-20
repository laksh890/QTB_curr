"""Polars query API — single source of truth for downstream models."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from iqrp.app.config.settings import AppSettings
from iqrp.app.data.storage.duckdb_catalog import DuckDBCatalog
from iqrp.app.data.storage.parquet_store import ParquetStore
from iqrp.app.data.types import MarketDataType


class MarketDataQueryService:
    """Read APIs returning Polars DataFrames exclusively."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        store: ParquetStore | None = None,
        catalog: DuckDBCatalog | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or ParquetStore(
            settings.storage.parquet_dir,
            compression=settings.data.ingestion.parquet_compression,
        )
        self.catalog = catalog

    def get_candles(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        return self.store.read(
            MarketDataType.CANDLE,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            timestamp_column="open_time",
        )

    def get_trades(
        self,
        *,
        exchange: str,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        return self.store.read(
            MarketDataType.TRADE,
            exchange=exchange,
            symbol=symbol,
            start=start,
            end=end,
            timestamp_column="timestamp",
        )

    def get_orderbook(
        self,
        *,
        exchange: str,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        return self.store.read(
            MarketDataType.ORDERBOOK,
            exchange=exchange,
            symbol=symbol,
            start=start,
            end=end,
            timestamp_column="timestamp",
        )

    def get_funding(
        self,
        *,
        exchange: str,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        return self.store.read(
            MarketDataType.FUNDING,
            exchange=exchange,
            symbol=symbol,
            start=start,
            end=end,
            timestamp_column="timestamp",
        )

    def get_open_interest(
        self,
        *,
        exchange: str,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        return self.store.read(
            MarketDataType.OPEN_INTEREST,
            exchange=exchange,
            symbol=symbol,
            start=start,
            end=end,
            timestamp_column="timestamp",
        )

    def sql(self, query: str) -> pl.DataFrame:
        if self.catalog is None:
            self.catalog = DuckDBCatalog(self.settings.storage.duckdb_path, self.store)
            self.catalog.register_all()
        return self.catalog.sql(query)


def get_candles(
    settings: AppSettings,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    return MarketDataQueryService(settings).get_candles(
        exchange=exchange, symbol=symbol, timeframe=timeframe, start=start, end=end
    )


def get_trades(
    settings: AppSettings,
    *,
    exchange: str,
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    return MarketDataQueryService(settings).get_trades(
        exchange=exchange, symbol=symbol, start=start, end=end
    )


def get_orderbook(
    settings: AppSettings,
    *,
    exchange: str,
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    return MarketDataQueryService(settings).get_orderbook(
        exchange=exchange, symbol=symbol, start=start, end=end
    )


def get_funding(
    settings: AppSettings,
    *,
    exchange: str,
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    return MarketDataQueryService(settings).get_funding(
        exchange=exchange, symbol=symbol, start=start, end=end
    )


def get_open_interest(
    settings: AppSettings,
    *,
    exchange: str,
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    return MarketDataQueryService(settings).get_open_interest(
        exchange=exchange, symbol=symbol, start=start, end=end
    )
