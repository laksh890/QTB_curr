"""Gap-only repair: download missing candles without full redownload."""

from __future__ import annotations

from datetime import datetime

import polars as pl
from loguru import logger

from iqrp.app.data.exchange.base import BaseExchange
from iqrp.app.data.ingestion.historical import HistoricalIngestor
from iqrp.app.data.storage.duckdb_catalog import DuckDBCatalog
from iqrp.app.data.storage.parquet_store import ParquetStore
from iqrp.app.data.types import MarketDataType, Timeframe
from iqrp.app.data.validation.validator import DataValidator


class DataRepair:
    """Repair candle gaps by fetching only missing ranges."""

    def __init__(
        self,
        exchange: BaseExchange,
        store: ParquetStore,
        catalog: DuckDBCatalog | None,
        *,
        page_limit: int,
        max_retries: int,
        retry_delay: float,
        retry_backoff: float,
    ) -> None:
        self.exchange = exchange
        self.store = store
        self.catalog = catalog
        self.validator = DataValidator()
        self.ingestor = HistoricalIngestor(
            exchange,
            page_limit=page_limit,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff=retry_backoff,
        )

    async def repair_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        existing = self.store.read(
            MarketDataType.CANDLE,
            exchange=self.exchange.name,
            symbol=symbol,
            timeframe=str(timeframe),
            start=start,
            end=end,
            timestamp_column="open_time",
        )
        missing = self.validator.find_missing_ranges(
            existing, timeframe=str(timeframe), start=start, end=end
        )
        if not missing:
            logger.info(
                "repair_noop exchange={} symbol={} tf={}",
                self.exchange.name,
                symbol,
                timeframe,
            )
            return existing

        frames: list[pl.DataFrame] = [existing] if not existing.is_empty() else []
        for gap_start, gap_end in missing:
            logger.info(
                "repair_gap exchange={} symbol={} tf={} start={} end={}",
                self.exchange.name,
                symbol,
                timeframe,
                gap_start.isoformat(),
                gap_end.isoformat(),
            )
            candles = await self.ingestor.download_candles(
                symbol, timeframe, start=gap_start, end=gap_end
            )
            if not candles:
                continue
            frame = pl.DataFrame([c.to_row() for c in candles])
            written = self.store.write_frame(
                frame,
                data_type=MarketDataType.CANDLE,
                exchange=self.exchange.name,
                symbol=self.exchange.normalize_symbol(symbol),
                timeframe=str(timeframe),
                timestamp_column="open_time",
            )
            if self.catalog is not None and written:
                self.catalog.register_data_type(MarketDataType.CANDLE)
            frames.append(frame)

        if not frames:
            return pl.DataFrame()
        return (
            pl.concat(frames, how="diagonal_relaxed")
            .unique(subset=["open_time"], keep="last")
            .sort("open_time")
        )
