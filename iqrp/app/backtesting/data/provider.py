"""Data provider abstraction for local and future remote sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from iqrp.app.backtesting.data.adapter import DataAdapter
from iqrp.app.backtesting.data.csv_adapter import CSVAdapter
from iqrp.app.backtesting.data.dataset_validator import DataQualityReport, DatasetValidator
from iqrp.app.backtesting.data.metadata import DatasetMetadata
from iqrp.app.backtesting.data.parquet_adapter import ParquetAdapter

__all__ = ["DataProvider", "LocalFileProvider"]


class DataProvider(ABC):
    """Abstract provider of historical datasets (local or remote).

    Remote network downloads are intentionally not implemented here; subclasses
    that fetch remote data must be introduced explicitly and audited separately.
    """

    @abstractmethod
    def list_datasets(self) -> list[str]:
        """Return known dataset identifiers."""

    @abstractmethod
    def get_adapter(self, dataset_id: str, **kwargs: Any) -> DataAdapter:
        """Return a :class:`DataAdapter` for ``dataset_id``."""

    def load(self, dataset_id: str, **kwargs: Any) -> pd.DataFrame:
        return self.get_adapter(dataset_id, **kwargs).load()

    def metadata(self, dataset_id: str, **kwargs: Any) -> DatasetMetadata:
        return self.get_adapter(dataset_id, **kwargs).metadata()

    def validate(
        self,
        dataset_id: str,
        *,
        raise_on_critical: bool = False,
        **kwargs: Any,
    ) -> DataQualityReport:
        return self.get_adapter(dataset_id, **kwargs).validate(
            raise_on_critical=raise_on_critical
        )


class LocalFileProvider(DataProvider):
    """Resolve datasets from a local directory of CSV / Parquet / Arrow files."""

    def __init__(
        self,
        root: str | Path,
        *,
        validator: DatasetValidator | None = None,
        recursive: bool = True,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"data root not found: {self.root}")
        self.validator = validator or DatasetValidator()
        self.recursive = recursive
        self._index = self._build_index()

    def _build_index(self) -> dict[str, Path]:
        patterns = ("*.csv", "*.parquet", "*.pq", "*.feather", "*.arrow")
        paths: list[Path] = []
        for pat in patterns:
            if self.recursive:
                paths.extend(self.root.rglob(pat))
            else:
                paths.extend(self.root.glob(pat))
        index: dict[str, Path] = {}
        for p in sorted(paths):
            key = p.stem
            # Prefer unique relative keys when collisions occur
            if key in index:
                rel = str(p.relative_to(self.root)).replace("/", "__")
                key = Path(rel).stem if "__" in rel else f"{key}__{p.suffix.lstrip('.')}"
            index[key] = p
            # Also register relative path without suffix
            try:
                rel_key = str(p.relative_to(self.root).with_suffix(""))
                index.setdefault(rel_key.replace("\\", "/"), p)
            except ValueError:
                pass
        return index

    def list_datasets(self) -> list[str]:
        return sorted(self._index.keys())

    def refresh(self) -> None:
        self._index = self._build_index()

    def resolve_path(self, dataset_id: str) -> Path:
        if dataset_id in self._index:
            return self._index[dataset_id]
        candidate = Path(dataset_id)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        if candidate.exists():
            return candidate
        raise KeyError(f"unknown dataset_id: {dataset_id!r}")

    def get_adapter(self, dataset_id: str, **kwargs: Any) -> DataAdapter:
        path = self.resolve_path(dataset_id)
        suffix = path.suffix.lower()
        common = {
            "dataset_id": kwargs.pop("dataset_id", path.stem),
            "version": kwargs.pop("version", "1.0.0"),
            "validator": kwargs.pop("validator", self.validator),
            "normalize": kwargs.pop("normalize", True),
        }
        if suffix == ".csv":
            return CSVAdapter(path, source="csv", **common, **kwargs)
        if suffix in {".parquet", ".pq", ".feather", ".arrow"} or path.is_dir():
            return ParquetAdapter(path, source="parquet", **common, **kwargs)
        raise ValueError(f"unsupported file type for dataset {dataset_id!r}: {path}")

    def load_universe(
        self,
        dataset_id: str,
        instruments: Sequence[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        return self.get_adapter(dataset_id, **kwargs).load_universe(instruments)
