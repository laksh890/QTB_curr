"""Deterministic local cache for historical provider requests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CacheKey:
    provider: str
    instrument: str
    start: str
    end: str
    frequency: str
    adjustment_policy: str

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class HistoricalCache:
    """Filesystem cache keyed by exact request identity."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else Path("data/cache/historical")
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: CacheKey) -> tuple[Path, Path]:
        d = self.root / key.provider / key.digest()
        return d / "data.parquet", d / "meta.json"

    def get(self, key: CacheKey) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        data_path, meta_path = self._paths(key)
        if not data_path.exists() or not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("cache_key") != asdict(key):
            return None
        frame = pd.read_parquet(data_path)
        return frame, meta

    def put(
        self,
        key: CacheKey,
        frame: pd.DataFrame,
        *,
        extra_meta: dict[str, Any] | None = None,
    ) -> Path:
        data_path, meta_path = self._paths(key)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = data_path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(data_path)
        meta = {
            "cache_key": asdict(key),
            "digest": key.digest(),
            "cached_at": datetime.now(UTC).isoformat(),
            "row_count": int(len(frame)),
            **dict(extra_meta or {}),
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return data_path


__all__ = ["CacheKey", "HistoricalCache"]
