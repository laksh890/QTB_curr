"""Historical dataset container with timestamp iteration and filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from iqrp.app.backtesting.data.adapter import DataAdapter
from iqrp.app.backtesting.data.dataset_validator import DataQualityReport, DatasetValidator
from iqrp.app.backtesting.data.metadata import DatasetMetadata, metadata_from_frame
from iqrp.app.backtesting.data.schema import normalize_frame

__all__ = ["HistoricalDataset", "create_synthetic_ohlcv"]


@dataclass
class HistoricalDataset:
    """In-memory historical OHLCV container for research / backtesting.

    Supports iteration by timestamp and instrument filtering. Timestamps are
    expected to be timezone-aware UTC after normalization.
    """

    frame: pd.DataFrame
    metadata: DatasetMetadata = field(default_factory=lambda: DatasetMetadata(dataset_id="unnamed"))
    quality_report: DataQualityReport | None = None

    def __post_init__(self) -> None:
        if self.frame is None:
            raise ValueError("frame is required")
        object.__setattr__(self, "frame", normalize_frame(self.frame))
        if self.metadata.row_count == 0 and not self.frame.empty:
            object.__setattr__(
                self,
                "metadata",
                metadata_from_frame(
                    self.frame,
                    dataset_id=self.metadata.dataset_id,
                    version=self.metadata.version,
                    source=self.metadata.source,
                    path=self.metadata.path,
                    checksum=self.metadata.checksum,
                    known_limitations=self.metadata.known_limitations,
                    instrument_metadata=self.metadata.instrument_metadata,
                ),
            )

    @classmethod
    def from_adapter(
        cls,
        adapter: DataAdapter,
        *,
        validate: bool = True,
        raise_on_critical: bool = False,
    ) -> HistoricalDataset:
        frame = adapter.load()
        meta = adapter.metadata()
        report = None
        if validate:
            report = adapter.validate(frame, raise_on_critical=raise_on_critical)
        return cls(frame=frame, metadata=meta, quality_report=report)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        dataset_id: str = "unnamed",
        version: str = "1.0.0",
        source: str = "local",
        validate: bool = False,
        validator: DatasetValidator | None = None,
    ) -> HistoricalDataset:
        meta = metadata_from_frame(
            normalize_frame(frame),
            dataset_id=dataset_id,
            version=version,
            source=source,
        )
        report = None
        if validate:
            report = (validator or DatasetValidator()).validate(
                frame, metadata=meta, normalize=True
            )
        return cls(frame=frame, metadata=meta, quality_report=report)

    def __len__(self) -> int:
        return int(len(self.frame))

    def __iter__(self) -> Iterator[tuple[datetime, pd.DataFrame]]:
        return self.iter_timestamps()

    @property
    def instruments(self) -> list[str]:
        if self.frame.empty:
            return []
        return sorted(self.frame["instrument"].astype(str).unique().tolist())

    @property
    def timestamps(self) -> list[datetime]:
        if self.frame.empty:
            return []
        ts = self.frame["timestamp"].drop_duplicates().sort_values()
        return [pd.Timestamp(t).to_pydatetime() for t in ts]

    def filter_instruments(self, instruments: Sequence[str]) -> HistoricalDataset:
        wanted = {str(i) for i in instruments}
        frame = self.frame.loc[self.frame["instrument"].astype(str).isin(wanted)].copy()
        meta = metadata_from_frame(
            frame,
            dataset_id=self.metadata.dataset_id,
            version=self.metadata.version,
            source=self.metadata.source,
            path=self.metadata.path,
            checksum=self.metadata.checksum,
            known_limitations=self.metadata.known_limitations,
        )
        return HistoricalDataset(frame=frame, metadata=meta, quality_report=self.quality_report)

    def filter_range(
        self,
        start: datetime | str | pd.Timestamp | None = None,
        end: datetime | str | pd.Timestamp | None = None,
    ) -> HistoricalDataset:
        frame = self.frame
        if start is not None:
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")
            frame = frame.loc[frame["timestamp"] >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            else:
                end_ts = end_ts.tz_convert("UTC")
            frame = frame.loc[frame["timestamp"] <= end_ts]
        frame = frame.reset_index(drop=True)
        meta = metadata_from_frame(
            frame,
            dataset_id=self.metadata.dataset_id,
            version=self.metadata.version,
            source=self.metadata.source,
            path=self.metadata.path,
            known_limitations=self.metadata.known_limitations,
        )
        return HistoricalDataset(frame=frame, metadata=meta, quality_report=self.quality_report)

    def at(self, timestamp: datetime | str | pd.Timestamp) -> pd.DataFrame:
        """Return the cross-section of all instruments at ``timestamp``."""
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return self.frame.loc[self.frame["timestamp"] == ts].reset_index(drop=True)

    def iter_timestamps(self) -> Iterator[tuple[datetime, pd.DataFrame]]:
        """Yield ``(timestamp, cross_section_frame)`` in chronological order."""
        if self.frame.empty:
            return iter(())
        for ts, grp in self.frame.groupby("timestamp", sort=True):
            yield pd.Timestamp(ts).to_pydatetime(), grp.reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "row_count": len(self),
            "instruments": self.instruments,
            "quality_report": self.quality_report.to_dict() if self.quality_report else None,
        }


def create_synthetic_ohlcv(
    *,
    n_days: int = 30,
    instruments: Sequence[str] | None = None,
    seed: int = 1,
    start: str | datetime | pd.Timestamp = "2020-01-01",
    freq: str = "1d",
) -> pd.DataFrame:
    """Deterministic synthetic OHLCV frame for fixtures (not investment advice)."""
    from iqrp.app.backtesting.data.synthetic import generate_synthetic_ohlcv

    return generate_synthetic_ohlcv(
        n_days=n_days,
        instruments=instruments,
        seed=seed,
        start=start,
        freq=freq,
    )
