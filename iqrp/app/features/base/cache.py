"""In-memory / disk feature computation cache."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0


@dataclass
class FeatureCache:
    """Cache computed feature frames keyed by content hash."""

    directory: Path | None = None
    max_entries: int = 256
    stats: CacheStats = field(default_factory=CacheStats)
    _store: dict[str, pl.DataFrame] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(
        feature_name: str,
        version: str,
        parameters: dict[str, Any],
        frame: pl.DataFrame,
        *,
        columns: tuple[str, ...] | None = None,
    ) -> str:
        cols = list(columns) if columns else sorted(frame.columns)
        subset = frame.select([c for c in cols if c in frame.columns])
        # Hash schema + row count + first/last timestamps if present for speed.
        digest_parts = {
            "feature": feature_name,
            "version": version,
            "parameters": parameters,
            "columns": cols,
            "height": subset.height,
            "width": subset.width,
            "dtypes": [str(t) for t in subset.dtypes],
        }
        if "open_time" in subset.columns and subset.height:
            digest_parts["first"] = str(subset["open_time"][0])
            digest_parts["last"] = str(subset["open_time"][-1])
            digest_parts["sum_close"] = (
                float(subset["close"].sum()) if "close" in subset.columns else 0.0
            )
        raw = json.dumps(digest_parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> pl.DataFrame | None:
        with self._lock:
            if key in self._store:
                self.stats.hits += 1
                return self._store[key]
            if self.directory is not None:
                path = self.directory / f"{key}.parquet"
                if path.exists():
                    frame = pl.read_parquet(path)
                    self._store[key] = frame
                    self.stats.hits += 1
                    return frame
            self.stats.misses += 1
            return None

    def put(self, key: str, frame: pl.DataFrame) -> None:
        with self._lock:
            if len(self._store) >= self.max_entries and key not in self._store:
                # Drop arbitrary oldest key (insertion order).
                oldest = next(iter(self._store))
                self._store.pop(oldest, None)
            self._store[key] = frame
            self.stats.stores += 1
            if self.directory is not None:
                path = self.directory / f"{key}.parquet"
                frame.write_parquet(path, compression="zstd")

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.stats = CacheStats()
            if self.directory is not None and self.directory.exists():
                for path in self.directory.glob("*.parquet"):
                    path.unlink(missing_ok=True)

    def snapshot_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "stores": self.stats.stores,
                "hit_rate": self.stats.hit_rate,
                "entries": len(self._store),
                "measured_at": time.time(),
            }
