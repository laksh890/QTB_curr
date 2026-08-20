"""Immutable registration helpers on top of DatasetRegistry (no silent overwrite)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iqrp.app.backtesting.data.dataset_registry import (
    DatasetRecord,
    DatasetRegistry,
    compute_checksum,
)
from iqrp.app.backtesting.data.metadata import metadata_from_frame
from iqrp.app.backtesting.data.schema import frame_coverage
from iqrp.app.data.historical.provenance import DatasetProvenance


class DatasetImmutabilityError(ValueError):
    """Raised when attempting to overwrite an existing id@version."""


def register_immutable(
    registry: DatasetRegistry,
    *,
    path: str | Path,
    dataset_id: str,
    version: str,
    source: str,
    frame=None,
    provenance: DatasetProvenance | None = None,
    quality_status: str | None = None,
    known_limitations: list[str] | None = None,
    persist: bool = True,
) -> DatasetRecord:
    """Register a dataset; refuse silent overwrite of the same id@version."""
    key = f"{dataset_id}@{version}"
    existing = registry.get(dataset_id, version)
    if existing is not None:
        raise DatasetImmutabilityError(
            f"dataset {key} already registered; create a new version or revision "
            f"(refusing silent overwrite)"
        )

    p = Path(path)
    checksum = compute_checksum(p)
    if frame is None:
        import pandas as pd

        frame = pd.read_parquet(p)

    meta = metadata_from_frame(
        frame,
        dataset_id=dataset_id,
        version=version,
        source=source,
        path=str(p),
    )
    cov = frame_coverage(frame, frequency=meta.frequency)
    rec = DatasetRecord.from_metadata(
        meta, path=p, checksum=checksum, coverage_pct=float(cov.get("coverage_pct", 0.0))
    )
    lim = list(known_limitations or [])
    if provenance is not None:
        lim.extend(provenance.known_limitations)
        rec.extra["provenance"] = provenance.to_dict()
        rec.extra["frequency_kind"] = provenance.frequency_kind
        rec.extra["data_class"] = provenance.data_class
        rec.extra["license_status"] = provenance.license_status
        if provenance.checksum:
            # keep file checksum as primary; store provenance checksum separately
            rec.extra["provenance_checksum_field"] = provenance.checksum
    if quality_status:
        rec.extra["quality_status"] = quality_status
    rec.known_limitations = sorted(set(lim + list(rec.known_limitations)))
    # update provenance checksum to file checksum
    if provenance is not None:
        provenance.checksum = checksum
        rec.extra["provenance"] = provenance.to_dict()

    return registry.register(rec, persist=persist)


def next_version(registry: DatasetRegistry, dataset_id: str, *, base: str = "1.0.0") -> str:
    """Suggest next patch version if base exists."""
    existing = [r.version for r in registry.list(dataset_id)]
    if base not in existing and f"{dataset_id}@{base}" not in {r.key for r in registry.list()}:
        return base
    # simple patch bump
    parts = base.split(".")
    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:  # noqa: BLE001
        return f"{base}.1"
    patch += 1
    while f"{major}.{minor}.{patch}" in existing:
        patch += 1
    return f"{major}.{minor}.{patch}"


__all__ = [
    "DatasetImmutabilityError",
    "next_version",
    "register_immutable",
]
