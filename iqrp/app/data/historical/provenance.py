"""Dataset provenance and DEVELOPMENT vs PRODUCTION labeling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def data_class_label(*, institutional: bool = False) -> str:
    return "PRODUCTION / INSTITUTIONAL DATA" if institutional else "DEVELOPMENT DATA"


@dataclass
class DatasetProvenance:
    """Full provenance record for an acquired / derived historical dataset."""

    provider: str
    source: str
    acquisition_timestamp: str
    original_symbol: str
    normalized_symbol: str
    frequency: str
    timezone: str = "UTC"
    original_timezone: str = "UNKNOWN"
    exchange_timezone: str = "Asia/Kolkata"
    currency: str = "INR"
    adjustment_status: str = "unadjusted"
    corporate_action_treatment: str = "UNKNOWN"
    contract_information: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    schema_version: str = "ohlcv_canonical_v1"
    license_status: str = "UNKNOWN"
    data_class: str = "DEVELOPMENT DATA"
    availability_timestamp_available: bool = False
    # derived dataset fields
    source_dataset_id: str | None = None
    source_frequency: str | None = None
    derived_frequency: str | None = None
    aggregation_method: str | None = None
    creation_timestamp: str | None = None
    frequency_kind: str = "SOURCE"  # SOURCE | DERIVED
    requested_range: tuple[str, str] | list[str] | None = None
    actual_range: tuple[str | None, str | None] | list[str | None] | None = None
    known_limitations: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if isinstance(d.get("requested_range"), tuple):
            d["requested_range"] = list(d["requested_range"])
        if isinstance(d.get("actual_range"), tuple):
            d["actual_range"] = list(d["actual_range"])
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetProvenance:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


# Futures extensibility keys (schema documentation — not continuous futures)
FUTURES_CONTRACT_FIELDS: tuple[str, ...] = (
    "symbol",
    "root_symbol",
    "expiry",
    "contract_month",
    "contract_type",
    "multiplier",
    "tick_size",
    "currency",
)


__all__ = [
    "FUTURES_CONTRACT_FIELDS",
    "DatasetProvenance",
    "data_class_label",
    "now_utc_iso",
]
