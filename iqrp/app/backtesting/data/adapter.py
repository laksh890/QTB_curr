"""Abstract historical data adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Sequence

import pandas as pd

from iqrp.app.backtesting.data.dataset_validator import DataQualityReport, DatasetValidator
from iqrp.app.backtesting.data.metadata import DatasetMetadata, metadata_from_frame
from iqrp.app.backtesting.data.schema import normalize_frame

__all__ = ["DataAdapter"]


class DataAdapter(ABC):
    """Common interface for loading and inspecting historical market data.

    Implementations must not download remote market data unless a dedicated
    remote :class:`~iqrp.app.backtesting.data.provider.DataProvider` is used.
    """

    def __init__(
        self,
        *,
        dataset_id: str | None = None,
        version: str = "1.0.0",
        source: str = "local",
        validator: DatasetValidator | None = None,
        normalize: bool = True,
    ) -> None:
        self.dataset_id = dataset_id or "unnamed"
        self.version = version
        self.source = source
        self.validator = validator or DatasetValidator()
        self.normalize = normalize
        self._cached: pd.DataFrame | None = None
        self._metadata: DatasetMetadata | None = None

    @abstractmethod
    def _read_raw(self) -> pd.DataFrame:
        """Load the underlying dataset without caching."""

    def load(self, *, refresh: bool = False) -> pd.DataFrame:
        """Load (and optionally normalize) the full dataset."""
        if self._cached is None or refresh:
            raw = self._read_raw()
            self._cached = normalize_frame(raw) if self.normalize else raw
            self._metadata = None
        return self._cached.copy()

    def validate(
        self,
        frame: pd.DataFrame | None = None,
        *,
        raise_on_critical: bool = False,
    ) -> DataQualityReport:
        """Validate ``frame`` (or the loaded dataset) and return a quality report."""
        data = frame if frame is not None else self.load()
        return self.validator.validate(
            data,
            metadata=self.metadata(),
            normalize=False,
            raise_on_critical=raise_on_critical,
        )

    def metadata(self, *, refresh: bool = False) -> DatasetMetadata:
        """Return dataset metadata derived from the loaded frame."""
        if self._metadata is None or refresh:
            frame = self.load(refresh=refresh)
            self._metadata = metadata_from_frame(
                frame,
                dataset_id=self.dataset_id,
                version=self.version,
                source=self.source,
                path=str(getattr(self, "path", "") or "") or None,
            )
        return self._metadata

    def available_instruments(self) -> list[str]:
        """Return sorted unique instrument identifiers."""
        frame = self.load()
        if frame.empty:
            return []
        return sorted(frame["instrument"].astype(str).unique().tolist())

    def available_dates(self) -> list[datetime]:
        """Return sorted unique timestamps as Python datetimes."""
        frame = self.load()
        if frame.empty:
            return []
        ts = frame["timestamp"].drop_duplicates().sort_values()
        return [pd.Timestamp(t).to_pydatetime() for t in ts]

    def load_range(
        self,
        start: datetime | str | pd.Timestamp | None = None,
        end: datetime | str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Load rows with ``start <= timestamp <= end`` (inclusive, UTC)."""
        frame = self.load()
        if frame.empty:
            return frame.copy()
        out = frame
        if start is not None:
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            else:
                start_ts = start_ts.tz_convert("UTC")
            out = out.loc[out["timestamp"] >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            else:
                end_ts = end_ts.tz_convert("UTC")
            out = out.loc[out["timestamp"] <= end_ts]
        return out.reset_index(drop=True)

    def load_instrument(self, instrument: str) -> pd.DataFrame:
        """Load all rows for a single instrument."""
        frame = self.load()
        if frame.empty:
            return frame.copy()
        return (
            frame.loc[frame["instrument"].astype(str) == str(instrument)]
            .reset_index(drop=True)
        )

    def load_universe(self, instruments: Sequence[str]) -> pd.DataFrame:
        """Load rows for a collection of instruments."""
        frame = self.load()
        if frame.empty:
            return frame.copy()
        wanted = {str(i) for i in instruments}
        return (
            frame.loc[frame["instrument"].astype(str).isin(wanted)]
            .reset_index(drop=True)
        )

    def clear_cache(self) -> None:
        self._cached = None
        self._metadata = None
