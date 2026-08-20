"""Partitioned Apache Parquet store (ZSTD, Hive-style layout)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
from loguru import logger

from iqrp.app.core.exceptions import DataError
from iqrp.app.data.types import MarketDataType

Compression = Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"]


class ParquetStore:
    """Write/read market-data frames with exchange/symbol/timeframe/day partitions."""

    def __init__(self, root: Path, *, compression: str = "zstd") -> None:
        self.root = Path(root)
        self.compression: Compression = cast(Compression, compression)
        self.root.mkdir(parents=True, exist_ok=True)

    def partition_dir(
        self,
        data_type: MarketDataType | str,
        *,
        exchange: str,
        symbol: str,
        timeframe: str | None,
        ts: datetime,
    ) -> Path:
        path = (
            self.root
            / str(data_type)
            / f"exchange={exchange.lower()}"
            / f"symbol={symbol.upper() if '-' not in symbol else symbol}"
        )
        if timeframe:
            path = path / f"timeframe={timeframe}"
        return path / f"year={ts.year:04d}" / f"month={ts.month:02d}" / f"day={ts.day:02d}"

    def write_frame(
        self,
        frame: pl.DataFrame,
        *,
        data_type: MarketDataType | str,
        exchange: str,
        symbol: str,
        timeframe: str | None,
        timestamp_column: str,
    ) -> list[Path]:
        """Stream-partition ``frame`` to disk; returns written file paths."""
        if frame.is_empty():
            return []
        if timestamp_column not in frame.columns:
            raise DataError(
                f"Missing timestamp column '{timestamp_column}'",
                code="PARQUET_MISSING_TS",
            )

        written: list[Path] = []
        # Ensure datetime for partitioning.
        ts_series = frame[timestamp_column]
        work = frame.with_columns(
            pl.col(timestamp_column).dt.year().alias("_year"),
            pl.col(timestamp_column).dt.month().alias("_month"),
            pl.col(timestamp_column).dt.day().alias("_day"),
        )
        keys = work.select("_year", "_month", "_day").unique().sort("_year", "_month", "_day")
        for row in keys.iter_rows(named=True):
            part = work.filter(
                (pl.col("_year") == row["_year"])
                & (pl.col("_month") == row["_month"])
                & (pl.col("_day") == row["_day"])
            ).drop(["_year", "_month", "_day"])
            sample_ts = part[timestamp_column][0]
            if not isinstance(sample_ts, datetime):
                sample_ts = datetime.fromisoformat(str(sample_ts))
            directory = self.partition_dir(
                data_type,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                ts=sample_ts,
            )
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "data.parquet"
            if path.exists():
                existing = pl.read_parquet(path)
                part = (
                    pl.concat([existing, part], how="diagonal_relaxed")
                    .unique(subset=[timestamp_column], keep="last")
                    .sort(timestamp_column)
                )
            part.write_parquet(path, compression=self.compression)
            written.append(path)
            logger.info(
                "parquet_write path={} rows={} compression={}",
                path,
                part.height,
                self.compression,
            )
        del ts_series
        return written

    def read(
        self,
        data_type: MarketDataType | str,
        *,
        exchange: str,
        symbol: str,
        timeframe: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        timestamp_column: str = "open_time",
    ) -> pl.DataFrame:
        """Read and optionally time-filter a partitioned dataset."""
        base = self.root / str(data_type) / f"exchange={exchange.lower()}"
        symbol_key = symbol.upper() if "-" not in symbol else symbol
        base = base / f"symbol={symbol_key}"
        if timeframe:
            base = base / f"timeframe={timeframe}"
        if not base.exists():
            return pl.DataFrame()
        files = sorted(base.rglob("data.parquet"))
        if not files:
            return pl.DataFrame()
        frame = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
        if timestamp_column in frame.columns:
            frame = frame.sort(timestamp_column)
            if start is not None:
                frame = frame.filter(pl.col(timestamp_column) >= start)
            if end is not None:
                frame = frame.filter(pl.col(timestamp_column) <= end)
        return frame

    def list_parquet_files(self, data_type: MarketDataType | str | None = None) -> list[Path]:
        root = self.root if data_type is None else self.root / str(data_type)
        if not root.exists():
            return []
        return sorted(root.rglob("*.parquet"))

    def storage_stats(self) -> dict[str, Any]:
        files = self.list_parquet_files()
        total_bytes = sum(f.stat().st_size for f in files)
        stats = {"file_count": len(files), "total_bytes": total_bytes}
        logger.info("storage_stats {}", stats)
        return stats
