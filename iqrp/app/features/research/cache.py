"""Simple disk cache for expensive research matrices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl


class ResearchCache:
    def __init__(self, directory: Path | None, *, enabled: bool = True) -> None:
        self.enabled = enabled and directory is not None
        self.directory = directory
        if self.enabled and self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(tag: str, frame: pl.DataFrame, columns: list[str], params: dict[str, Any]) -> str:
        payload = {
            "tag": tag,
            "columns": columns,
            "height": frame.height,
            "params": params,
        }
        if "open_time" in frame.columns and frame.height:
            payload["first"] = str(frame["open_time"][0])
            payload["last"] = str(frame["open_time"][-1])
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_frame(self, key: str) -> pl.DataFrame | None:
        if not self.enabled or self.directory is None:
            return None
        path = self.directory / f"{key}.parquet"
        if path.exists():
            return pl.read_parquet(path)
        return None

    def put_frame(self, key: str, frame: pl.DataFrame) -> None:
        if not self.enabled or self.directory is None:
            return
        frame.write_parquet(self.directory / f"{key}.parquet", compression="zstd")

    def get_json(self, key: str) -> dict[str, Any] | None:
        if not self.enabled or self.directory is None:
            return None
        path = self.directory / f"{key}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        return None

    def put_json(self, key: str, payload: dict[str, Any]) -> None:
        if not self.enabled or self.directory is None:
            return
        (self.directory / f"{key}.json").write_text(
            json.dumps(payload, default=str), encoding="utf-8"
        )
