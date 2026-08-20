"""Candle / series validation and quality reporting."""

from __future__ import annotations

from datetime import datetime

import polars as pl
from loguru import logger

from iqrp.app.data.models import DataQualityReport
from iqrp.app.data.types import timeframe_to_timedelta
from iqrp.app.data.validation.anomalies import Anomaly, AnomalyKind


class DataValidator:
    """Detect structural and economic anomalies in OHLCV frames."""

    REQUIRED_CANDLE_COLUMNS = (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    def validate_candles(
        self,
        frame: pl.DataFrame,
        *,
        timeframe: str,
        exchange: str,
        symbol: str,
        exchange_latency_ms: float | None = None,
    ) -> tuple[list[Anomaly], DataQualityReport]:
        anomalies: list[Anomaly] = []

        for col in self.REQUIRED_CANDLE_COLUMNS:
            if col not in frame.columns:
                anomalies.append(
                    Anomaly(
                        kind=AnomalyKind.MISSING_FIELD,
                        message=f"Missing field '{col}'",
                    )
                )
        if anomalies:
            report = DataQualityReport(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                data_type="candles",
                row_count=frame.height,
                missing_pct=100.0,
                coverage_pct=0.0,
                issues=tuple(a.message for a in anomalies),
                exchange_latency_ms=exchange_latency_ms,
            )
            return anomalies, report

        if frame.is_empty():
            report = DataQualityReport(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                data_type="candles",
                row_count=0,
                missing_pct=100.0,
                coverage_pct=0.0,
                issues=("empty frame",),
                exchange_latency_ms=exchange_latency_ms,
            )
            return anomalies, report

        raw_times = frame["open_time"].to_list()
        for idx in range(1, len(raw_times)):
            if raw_times[idx] < raw_times[idx - 1]:
                anomalies.append(
                    Anomaly(
                        kind=AnomalyKind.INCORRECT_ORDER,
                        message="Timestamps not ascending",
                        timestamp=raw_times[idx],
                    )
                )
                break

        ordered = frame.sort("open_time")

        # Duplicates
        dup_count = ordered.height - ordered.unique(subset=["open_time"]).height
        if dup_count:
            anomalies.append(
                Anomaly(
                    kind=AnomalyKind.DUPLICATE_CANDLE,
                    message=f"{dup_count} duplicate open_time values",
                    details={"count": dup_count},
                )
            )

        # Negative volume / impossible OHLC
        bad_vol = ordered.filter(pl.col("volume") < 0)
        for row in bad_vol.iter_rows(named=True):
            anomalies.append(
                Anomaly(
                    kind=AnomalyKind.NEGATIVE_VOLUME,
                    message="Negative volume",
                    timestamp=row["open_time"],
                )
            )
        bad_ohlc = ordered.filter(
            (pl.col("high") < pl.col("low"))
            | (pl.col("open") > pl.col("high"))
            | (pl.col("open") < pl.col("low"))
            | (pl.col("close") > pl.col("high"))
            | (pl.col("close") < pl.col("low"))
        )
        for row in bad_ohlc.iter_rows(named=True):
            anomalies.append(
                Anomaly(
                    kind=AnomalyKind.IMPOSSIBLE_OHLC,
                    message="Impossible OHLC relationship",
                    timestamp=row["open_time"],
                )
            )

        # Gaps / missing candles
        step = timeframe_to_timedelta(timeframe)
        gap_starts: list[datetime] = []
        unique_times = ordered.unique(subset=["open_time"]).sort("open_time")["open_time"].to_list()
        for idx in range(1, len(unique_times)):
            prev = unique_times[idx - 1]
            curr = unique_times[idx]
            delta = curr - prev
            if delta > step:
                missing = int(delta / step) - 1
                gap_starts.append(prev)
                anomalies.append(
                    Anomaly(
                        kind=AnomalyKind.TIMESTAMP_GAP,
                        message=f"Gap of {missing} candles after {prev.isoformat()}",
                        timestamp=prev,
                        details={"missing": missing, "next": curr.isoformat()},
                    )
                )
                for _ in range(missing):
                    anomalies.append(
                        Anomaly(
                            kind=AnomalyKind.MISSING_CANDLE,
                            message="Missing candle in gap",
                            timestamp=prev,
                        )
                    )

        oldest = unique_times[0]
        newest = unique_times[-1]
        expected = int((newest - oldest) / step) + 1
        actual = len(unique_times)
        missing_pct = max(0.0, (expected - actual) / expected * 100.0) if expected else 0.0
        coverage = (actual / expected * 100.0) if expected else 100.0

        report = DataQualityReport(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            data_type="candles",
            row_count=actual,
            missing_pct=missing_pct,
            duplicate_count=int(dup_count),
            gap_count=len(gap_starts),
            coverage_pct=coverage,
            oldest_record=oldest,
            newest_record=newest,
            exchange_latency_ms=exchange_latency_ms,
            issues=tuple(a.message for a in anomalies[:50]),
        )
        if anomalies:
            logger.warning(
                "validation_failures exchange={} symbol={} tf={} count={}",
                exchange,
                symbol,
                timeframe,
                len(anomalies),
            )
        return anomalies, report

    def find_missing_ranges(
        self,
        frame: pl.DataFrame,
        *,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Return inclusive missing ranges within ``[start, end]``."""
        step = timeframe_to_timedelta(timeframe)
        if frame.is_empty() or "open_time" not in frame.columns:
            return [(start, end)]

        present = set(frame.sort("open_time")["open_time"].to_list())
        ranges: list[tuple[datetime, datetime]] = []
        cursor = start
        gap_start: datetime | None = None
        while cursor <= end:
            if cursor not in present:
                if gap_start is None:
                    gap_start = cursor
            elif gap_start is not None:
                ranges.append((gap_start, cursor - step))
                gap_start = None
            cursor = cursor + step
        if gap_start is not None:
            ranges.append((gap_start, end))
        return ranges
