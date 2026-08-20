"""Parquet / Arrow-compatible adapter for historical OHLCV datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from iqrp.app.backtesting.data.adapter import DataAdapter
from iqrp.app.backtesting.data.dataset_validator import DatasetValidator

__all__ = ["ParquetAdapter", "file_sha256", "parquet_canonical_sha256"]


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of raw file bytes."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parquet_canonical_sha256(path: str | Path) -> str:
    """SHA-256 of canonicalized parquet bytes (column-sorted table write)."""
    import hashlib
    import io

    table = pq.read_table(path)
    names = sorted(table.column_names)
    table = table.select(names)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="none")
    return hashlib.sha256(buf.getvalue()).hexdigest()


class ParquetAdapter(DataAdapter):
    """Load historical market data from Parquet, Feather, or Arrow IPC files.

    Also accepts an in-memory :class:`pyarrow.Table` for Arrow-compatible flows.
    """

    def __init__(
        self,
        path: str | Path | pa.Table,
        *,
        dataset_id: str | None = None,
        version: str = "1.0.0",
        source: str = "parquet",
        validator: DatasetValidator | None = None,
        normalize: bool = True,
        columns: list[str] | None = None,
    ) -> None:
        self._table: pa.Table | None = None
        self.columns = columns
        if isinstance(path, pa.Table):
            self.path = Path("")
            self._table = path
            default_id = "arrow_table"
        else:
            self.path = Path(path)
            if not self.path.exists():
                raise FileNotFoundError(f"Parquet/Arrow path not found: {self.path}")
            default_id = self.path.stem
        super().__init__(
            dataset_id=dataset_id or default_id,
            version=version,
            source=source,
            validator=validator,
            normalize=normalize,
        )

    def _read_raw(self) -> pd.DataFrame:
        table = self._load_table()
        if self.columns is not None:
            missing = [c for c in self.columns if c not in table.column_names]
            if missing:
                # Allow alias names; select intersection then let normalize rename
                available = [c for c in self.columns if c in table.column_names]
                if available:
                    table = table.select(available)
            else:
                table = table.select(self.columns)
        return table.to_pandas()

    def _load_table(self) -> pa.Table:
        if self._table is not None:
            return self._table
        path = self.path
        if path.is_dir():
            # Hive / multi-file parquet dataset
            return pq.read_table(path)
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            return pq.read_table(path)
        if suffix in {".feather", ".arrow"}:
            try:
                return feather.read_table(path)
            except Exception:  # noqa: BLE001
                with path.open("rb") as f:
                    reader = ipc.open_file(f)
                    return reader.read_all()
        # Fallback: try parquet then feather
        try:
            return pq.read_table(path)
        except Exception:
            return feather.read_table(path)

    def checksum(self, *, canonical: bool = False) -> str:
        """Return a SHA-256 checksum of the underlying file or table."""
        if self._table is not None:
            import hashlib
            import io

            names = sorted(self._table.column_names)
            table = self._table.select(names)
            buf = io.BytesIO()
            pq.write_table(table, buf, compression="none")
            return hashlib.sha256(buf.getvalue()).hexdigest()
        if canonical and self.path.suffix.lower() in {".parquet", ".pq"}:
            return parquet_canonical_sha256(self.path)
        return file_sha256(self.path)
