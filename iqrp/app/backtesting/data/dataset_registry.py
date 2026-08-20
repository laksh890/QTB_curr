"""Persistent registry of historical datasets (id / version / coverage / checksum)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from iqrp.app.backtesting.data.metadata import DatasetMetadata
from iqrp.app.backtesting.serializer import to_jsonable

__all__ = [
    "DatasetRecord",
    "DatasetRegistry",
    "compute_checksum",
]


def compute_checksum(path: str | Path, *, canonical_parquet: bool = False) -> str:
    """SHA-256 of file bytes, or of canonicalized parquet bytes when requested."""
    p = Path(path)
    if canonical_parquet and p.suffix.lower() in {".parquet", ".pq"}:
        import io

        table = pq.read_table(p)
        names = sorted(table.column_names)
        table = table.select(names)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="none")
        return hashlib.sha256(buf.getvalue()).hexdigest()
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(slots=True)
class DatasetRecord:
    """Registry entry for a versioned historical dataset."""

    dataset_id: str
    version: str
    source: str
    path: str
    checksum: str
    frequency: str = "unknown"
    timezone: str = "UTC"
    start: str | None = None
    end: str | None = None
    instrument_count: int = 0
    row_count: int = 0
    instruments: list[str] = field(default_factory=list)
    coverage_pct: float | None = None
    corporate_actions_available: bool = False
    liquidity_data_available: bool = False
    known_limitations: list[str] = field(default_factory=list)
    registered_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.dataset_id}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    @classmethod
    def from_metadata(
        cls,
        metadata: DatasetMetadata,
        *,
        path: str | Path,
        checksum: str,
        coverage_pct: float | None = None,
    ) -> DatasetRecord:
        return cls(
            dataset_id=metadata.dataset_id,
            version=metadata.version,
            source=metadata.source,
            path=str(path),
            checksum=checksum,
            frequency=metadata.frequency,
            timezone=metadata.timezone,
            start=metadata.start.isoformat() if metadata.start else None,
            end=metadata.end.isoformat() if metadata.end else None,
            instrument_count=metadata.instrument_count,
            row_count=metadata.row_count,
            instruments=list(metadata.instruments),
            coverage_pct=coverage_pct,
            corporate_actions_available=metadata.corporate_actions_available,
            liquidity_data_available=metadata.liquidity_data_available,
            known_limitations=list(metadata.known_limitations),
            extra=dict(metadata.extra),
        )


class DatasetRegistry:
    """JSON-backed registry of dataset id / version / source / coverage / checksum."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else Path.cwd() / "dataset_registry.json"
        self._records: dict[str, DatasetRecord] = {}
        if self.path.exists():
            self.load()

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, key: str) -> bool:
        return key in self._records or any(
            r.dataset_id == key for r in self._records.values()
        )

    def register(
        self,
        record: DatasetRecord | DatasetMetadata,
        *,
        path: str | Path | None = None,
        checksum: str | None = None,
        coverage_pct: float | None = None,
        persist: bool = True,
    ) -> DatasetRecord:
        if isinstance(record, DatasetMetadata):
            if path is None and not record.path:
                raise ValueError("path is required when registering DatasetMetadata")
            p = Path(path or record.path)  # type: ignore[arg-type]
            cs = checksum or (record.checksum or compute_checksum(p))
            rec = DatasetRecord.from_metadata(
                record, path=p, checksum=cs, coverage_pct=coverage_pct
            )
        else:
            rec = record
        self._records[rec.key] = rec
        if persist:
            self.save()
        return rec

    def register_file(
        self,
        path: str | Path,
        *,
        dataset_id: str | None = None,
        version: str = "1.0.0",
        source: str = "local",
        canonical_parquet: bool = False,
        persist: bool = True,
    ) -> DatasetRecord:
        """Inspect a local CSV/Parquet file and register it."""
        from iqrp.app.backtesting.data.csv_adapter import CSVAdapter
        from iqrp.app.backtesting.data.parquet_adapter import ParquetAdapter
        from iqrp.app.backtesting.data.schema import frame_coverage

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() == ".csv":
            adapter = CSVAdapter(
                p, dataset_id=dataset_id or p.stem, version=version, source=source
            )
        else:
            adapter = ParquetAdapter(
                p, dataset_id=dataset_id or p.stem, version=version, source=source
            )
        frame = adapter.load()
        meta = adapter.metadata()
        cov = frame_coverage(frame, frequency=meta.frequency)
        cs = compute_checksum(p, canonical_parquet=canonical_parquet)
        meta.checksum = cs
        return self.register(
            meta,
            path=p,
            checksum=cs,
            coverage_pct=float(cov.get("coverage_pct", 0.0)),
            persist=persist,
        )

    def get(self, dataset_id: str, version: str | None = None) -> DatasetRecord | None:
        if version is not None:
            return self._records.get(f"{dataset_id}@{version}")
        # latest by registered_at among matching ids
        matches = [r for r in self._records.values() if r.dataset_id == dataset_id]
        if not matches:
            # allow full key
            return self._records.get(dataset_id)
        matches.sort(key=lambda r: r.registered_at, reverse=True)
        return matches[0]

    def require(self, dataset_id: str, version: str | None = None) -> DatasetRecord:
        rec = self.get(dataset_id, version)
        if rec is None:
            raise KeyError(f"unknown dataset: {dataset_id}@{version or 'latest'}")
        return rec

    def list(self, dataset_id: str | None = None) -> list[DatasetRecord]:
        rows = list(self._records.values())
        if dataset_id is not None:
            rows = [r for r in rows if r.dataset_id == dataset_id]
        rows.sort(key=lambda r: (r.dataset_id, r.version, r.registered_at))
        return rows

    def remove(self, dataset_id: str, version: str, *, persist: bool = True) -> None:
        key = f"{dataset_id}@{version}"
        self._records.pop(key, None)
        if persist:
            self.save()

    def verify_checksum(self, dataset_id: str, version: str | None = None) -> bool:
        rec = self.require(dataset_id, version)
        current = compute_checksum(rec.path)
        if current == rec.checksum:
            return True
        # Registrations may store a canonical parquet digest; accept either form.
        p = Path(rec.path)
        if p.suffix.lower() in {".parquet", ".pq"}:
            return compute_checksum(p, canonical_parquet=True) == rec.checksum
        return False

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(UTC).isoformat(),
            "datasets": [r.to_dict() for r in self.list()],
        }
        self.path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
        return self.path

    def load(self) -> None:
        if not self.path.exists():
            self._records = {}
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        records = payload.get("datasets", payload if isinstance(payload, list) else [])
        self._records = {}
        for item in records:
            rec = DatasetRecord.from_dict(item)
            self._records[rec.key] = rec
