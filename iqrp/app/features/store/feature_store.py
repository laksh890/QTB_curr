"""Parquet feature store with incremental updates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger


class FeatureStore:
    """Persist feature frames partitioned by exchange/symbol/tf/group/year/month."""

    def __init__(self, root: Path, *, compression: str = "zstd") -> None:
        self.root = Path(root)
        self.compression = compression
        self.root.mkdir(parents=True, exist_ok=True)

    def _partition_dir(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        feature_group: str,
        ts: datetime,
    ) -> Path:
        return (
            self.root
            / f"exchange={exchange.lower()}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"feature_group={feature_group}"
            / f"year={ts.year:04d}"
            / f"month={ts.month:02d}"
        )

    def write(
        self,
        frame: pl.DataFrame,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        feature_group: str,
        timestamp_column: str = "open_time",
    ) -> list[Path]:
        if frame.is_empty():
            return []
        if timestamp_column not in frame.columns:
            from iqrp.app.core.exceptions import DataError

            raise DataError(
                f"Missing timestamp column '{timestamp_column}'",
                code="FEATURE_STORE_MISSING_TS",
            )
        work = frame.with_columns(
            pl.col(timestamp_column).dt.year().alias("_y"),
            pl.col(timestamp_column).dt.month().alias("_m"),
        )
        keys = work.select("_y", "_m").unique().sort("_y", "_m")
        written: list[Path] = []
        for row in keys.iter_rows(named=True):
            part = work.filter((pl.col("_y") == row["_y"]) & (pl.col("_m") == row["_m"])).drop(
                ["_y", "_m"]
            )
            ts = part[timestamp_column][0]
            if not isinstance(ts, datetime):
                ts = datetime.fromisoformat(str(ts))
            directory = self._partition_dir(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                feature_group=feature_group,
                ts=ts,
            )
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "features.parquet"
            if path.exists():
                existing = pl.read_parquet(path)
                part = (
                    pl.concat([existing, part], how="diagonal_relaxed")
                    .unique(subset=[timestamp_column], keep="last")
                    .sort(timestamp_column)
                )
            part.write_parquet(path, compression=self.compression)  # type: ignore[arg-type]
            written.append(path)
            logger.info("feature_store_write path={} rows={}", path, part.height)
        return written

    def read(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        feature_group: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        timestamp_column: str = "open_time",
    ) -> pl.DataFrame:
        base = (
            self.root
            / f"exchange={exchange.lower()}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
        )
        if feature_group:
            base = base / f"feature_group={feature_group}"
        if not base.exists():
            return pl.DataFrame()
        files = sorted(base.rglob("features.parquet"))
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

    def update_incremental(
        self,
        frame: pl.DataFrame,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        feature_group: str,
        timestamp_column: str = "open_time",
    ) -> list[Path]:
        """Write only rows newer than the latest stored timestamp."""
        existing = self.read(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            feature_group=feature_group,
            timestamp_column=timestamp_column,
        )
        if existing.is_empty() or timestamp_column not in existing.columns:
            return self.write(
                frame,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                feature_group=feature_group,
                timestamp_column=timestamp_column,
            )
        latest = existing[timestamp_column].max()
        new_rows = frame.filter(pl.col(timestamp_column) > latest)
        if new_rows.is_empty():
            logger.info("feature_store_incremental_noop")
            return []
        return self.write(
            new_rows,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            feature_group=feature_group,
            timestamp_column=timestamp_column,
        )

    def stats(self) -> dict[str, Any]:
        files = list(self.root.rglob("features.parquet"))
        return {"file_count": len(files), "total_bytes": sum(f.stat().st_size for f in files)}
