"""Incremental updater — append only new candles since last stored timestamp."""

from __future__ import annotations

from datetime import datetime

import polars as pl
from loguru import logger

from iqrp.app.common.datetime_utils import utc_now
from iqrp.app.config.settings import AppSettings
from iqrp.app.data.exchange.base import BaseExchange
from iqrp.app.data.services.downloader import DataDownloader
from iqrp.app.data.storage.duckdb_catalog import DuckDBCatalog
from iqrp.app.data.storage.parquet_store import ParquetStore
from iqrp.app.data.types import MarketDataType, Timeframe, timeframe_to_timedelta


class DataUpdater:
    """Incrementally extend stored candle history to ``end`` (default: now)."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        exchange: BaseExchange | None = None,
        store: ParquetStore | None = None,
        catalog: DuckDBCatalog | None = None,
    ) -> None:
        self.settings = settings
        self.downloader = DataDownloader(
            settings, exchange=exchange, store=store, catalog=catalog
        )
        self.store = self.downloader.store
        self.exchange = self.downloader.exchange

    async def update_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        end = end or utc_now()
        existing = self.store.read(
            MarketDataType.CANDLE,
            exchange=self.exchange.name,
            symbol=self.exchange.normalize_symbol(symbol),
            timeframe=str(timeframe),
            timestamp_column="open_time",
        )
        if existing.is_empty():
            raise ValueError(
                "No existing candles to update incrementally; run a full download first"
            )
        last = existing["open_time"].max()
        assert isinstance(last, datetime)
        start = last + timeframe_to_timedelta(timeframe)
        if start > end:
            logger.info("update_noop symbol={} tf={}", symbol, timeframe)
            return existing
        logger.info(
            "update_incremental symbol={} tf={} start={} end={}",
            symbol,
            timeframe,
            start.isoformat(),
            end.isoformat(),
        )
        new_frame = await self.downloader.download_candles(
            symbol, timeframe, start=start, end=end, resume=False
        )
        if new_frame.is_empty():
            return existing
        return (
            pl.concat([existing, new_frame], how="diagonal_relaxed")
            .unique(subset=["open_time"], keep="last")
            .sort("open_time")
        )
