"""Dataset and instrument metadata for historical backtesting data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

import pandas as pd

from iqrp.app.backtesting.data.schema import infer_frequency
from iqrp.app.backtesting.serializer import to_jsonable

__all__ = [
    "InstrumentMetadata",
    "DatasetMetadata",
    "CoverageInfo",
    "metadata_from_frame",
]


@dataclass(slots=True)
class InstrumentMetadata:
    """Per-instrument descriptive metadata (optional contract fields)."""

    instrument: str
    currency: str | None = None
    exchange: str | None = None
    tick_size: float | None = None
    tick_value: float | None = None
    multiplier: float | None = None
    margin: float | None = None
    contract: str | None = None
    expiry: datetime | None = None
    asset_class: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InstrumentMetadata:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known and k != "extra"}
        extra = dict(data.get("extra") or {})
        for k, v in data.items():
            if k not in known:
                extra[k] = v
        kwargs["extra"] = extra
        expiry = kwargs.get("expiry")
        if isinstance(expiry, str):
            kwargs["expiry"] = datetime.fromisoformat(expiry)
        return cls(**kwargs)


@dataclass(slots=True)
class CoverageInfo:
    """Temporal / instrument coverage summary."""

    start: datetime | None = None
    end: datetime | None = None
    frequency: str = "unknown"
    instrument_count: int = 0
    row_count: int = 0
    coverage_pct: float = 0.0
    instruments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(slots=True)
class DatasetMetadata:
    """Identifying and descriptive metadata for a historical dataset."""

    dataset_id: str
    version: str = "1.0.0"
    source: str = "local"
    path: str | None = None
    frequency: str = "unknown"
    timezone: str = "UTC"
    start: datetime | None = None
    end: datetime | None = None
    instrument_count: int = 0
    row_count: int = 0
    checksum: str | None = None
    instruments: list[str] = field(default_factory=list)
    instrument_metadata: dict[str, InstrumentMetadata] = field(default_factory=dict)
    corporate_actions_available: bool = False
    liquidity_data_available: bool = False
    known_limitations: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["instrument_metadata"] = {
            k: (v.to_dict() if isinstance(v, InstrumentMetadata) else v)
            for k, v in self.instrument_metadata.items()
        }
        return to_jsonable(d)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetMetadata:
        payload = dict(data)
        im_raw = payload.pop("instrument_metadata", {}) or {}
        instrument_metadata = {
            str(k): InstrumentMetadata.from_dict(v) if isinstance(v, Mapping) else v
            for k, v in im_raw.items()
        }
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in payload.items() if k in known}
        for key in ("start", "end"):
            val = kwargs.get(key)
            if isinstance(val, str):
                kwargs[key] = datetime.fromisoformat(val)
        kwargs["instrument_metadata"] = instrument_metadata
        return cls(**kwargs)

    @property
    def coverage(self) -> CoverageInfo:
        return CoverageInfo(
            start=self.start,
            end=self.end,
            frequency=self.frequency,
            instrument_count=self.instrument_count,
            row_count=self.row_count,
            instruments=list(self.instruments),
        )


def metadata_from_frame(
    frame: pd.DataFrame,
    *,
    dataset_id: str = "unnamed",
    version: str = "1.0.0",
    source: str = "local",
    path: str | None = None,
    checksum: str | None = None,
    known_limitations: Sequence[str] | None = None,
    instrument_metadata: Mapping[str, InstrumentMetadata | Mapping[str, Any]] | None = None,
) -> DatasetMetadata:
    """Build :class:`DatasetMetadata` from a normalized OHLCV frame."""
    if frame.empty:
        return DatasetMetadata(
            dataset_id=dataset_id,
            version=version,
            source=source,
            path=path,
            checksum=checksum,
            known_limitations=list(known_limitations or []),
        )

    instruments = sorted(frame["instrument"].astype(str).unique().tolist())
    freq = infer_frequency(frame["timestamp"])
    start = pd.Timestamp(frame["timestamp"].min()).to_pydatetime()
    end = pd.Timestamp(frame["timestamp"].max()).to_pydatetime()

    liq_cols = {"bid", "ask", "bid_size", "ask_size", "volume"}
    liquidity = bool(liq_cols.intersection(frame.columns))

    im: dict[str, InstrumentMetadata] = {}
    if instrument_metadata:
        for key, val in instrument_metadata.items():
            im[str(key)] = (
                val
                if isinstance(val, InstrumentMetadata)
                else InstrumentMetadata.from_dict(val)
            )
    else:
        for inst in instruments:
            sub = frame.loc[frame["instrument"] == inst]
            currency = None
            exchange = None
            contract = None
            expiry = None
            if "currency" in sub.columns and sub["currency"].notna().any():
                currency = str(sub["currency"].dropna().iloc[0])
            if "exchange" in sub.columns and sub["exchange"].notna().any():
                exchange = str(sub["exchange"].dropna().iloc[0])
            if "contract" in sub.columns and sub["contract"].notna().any():
                contract = str(sub["contract"].dropna().iloc[0])
            if "expiry" in sub.columns and sub["expiry"].notna().any():
                expiry = pd.Timestamp(sub["expiry"].dropna().iloc[0]).to_pydatetime()
            im[inst] = InstrumentMetadata(
                instrument=inst,
                currency=currency,
                exchange=exchange,
                contract=contract,
                expiry=expiry,
            )

    return DatasetMetadata(
        dataset_id=dataset_id,
        version=version,
        source=source,
        path=path,
        frequency=freq,
        timezone="UTC",
        start=start,
        end=end,
        instrument_count=len(instruments),
        row_count=int(len(frame)),
        checksum=checksum,
        instruments=instruments,
        instrument_metadata=im,
        corporate_actions_available=False,
        liquidity_data_available=liquidity,
        known_limitations=list(known_limitations or []),
    )
