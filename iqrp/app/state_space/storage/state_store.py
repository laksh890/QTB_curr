"""Persist state-space artifacts to Parquet + JSON + DuckDB."""

# ruff: noqa: S608

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
from loguru import logger

from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.base.forecast_result import ForecastResult
from iqrp.app.state_space.base.smoother_result import SmootherResult
from iqrp.app.state_space.config import StateSpaceSettings


class StateStore:
    """Hive-style parquet store with optional DuckDB registration."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        duckdb_path: Path | None = None,
        compression: str | None = None,
        register_duckdb: bool | None = None,
        settings: StateSpaceSettings | None = None,
    ) -> None:
        settings = settings or StateSpaceSettings.default()
        self.root = Path(root) if root else Path(settings.store_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.compression = compression or settings.storage.compression
        self.register_duckdb = (
            settings.storage.register_duckdb if register_duckdb is None else register_duckdb
        )
        self.duckdb_path = Path(duckdb_path) if duckdb_path else Path(settings.duckdb_path)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    def write_filter_result(
        self,
        result: FilterResult,
        *,
        model_name: str,
        version: str,
        exchange: str = "synthetic",
        symbol: str = "STATE",
        timeframe: str = "1h",
        timestamps: list[Any] | None = None,
        forecast: ForecastResult | None = None,
        diagnostics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        base = self._partition(exchange, symbol, timeframe, model_name, version)
        base.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        frame = result.to_frame(timestamps=timestamps)
        states_path = base / "states.parquet"
        frame.write_parquet(states_path, compression=self.compression)  # type: ignore[arg-type]
        paths["states"] = states_path

        proba_path = base / "probabilities.parquet"
        proba_data: dict[str, Any] = {}
        if timestamps is not None:
            proba_data["open_time"] = timestamps
        for j in range(result.n_states):
            proba_data[f"proba_{j}"] = result.filtered_probabilities[:, j].tolist()
        pl.DataFrame(proba_data).write_parquet(
            proba_path, compression=self.compression  # type: ignore[arg-type]
        )
        paths["probabilities"] = proba_path

        if forecast is not None:
            fc_path = base / "forecast.json"
            fc_path.write_text(json.dumps(forecast.to_dict(), indent=2), encoding="utf-8")
            paths["forecast"] = fc_path

        if diagnostics is not None:
            diag_path = base / "diagnostics.json"
            diag_path.write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
            paths["diagnostics"] = diag_path

        meta_path = base / "metadata.json"
        meta_path.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "version": version,
                    "log_likelihood": result.log_likelihood,
                    "metadata": metadata or {},
                    "written_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        paths["metadata"] = meta_path

        if self.register_duckdb:
            self._register_duckdb(base)

        logger.info("state_store_write base={} files={}", base, list(paths))
        return paths

    def write_smoother_result(
        self,
        result: SmootherResult,
        *,
        model_name: str,
        version: str,
        exchange: str = "synthetic",
        symbol: str = "STATE",
        timeframe: str = "1h",
        timestamps: list[Any] | None = None,
    ) -> dict[str, Path]:
        base = self._partition(exchange, symbol, timeframe, model_name, version)
        base.mkdir(parents=True, exist_ok=True)
        path = base / "smoothed_states.parquet"
        result.to_frame(timestamps=timestamps).write_parquet(
            path, compression=self.compression  # type: ignore[arg-type]
        )
        return {"smoothed_states": path}

    def write_transition_matrix(
        self,
        matrix: np.ndarray,
        *,
        model_name: str,
        version: str,
        exchange: str = "synthetic",
        symbol: str = "STATE",
        timeframe: str = "1h",
    ) -> Path:
        base = self._partition(exchange, symbol, timeframe, model_name, version)
        base.mkdir(parents=True, exist_ok=True)
        k = int(matrix.shape[0])
        tm = pl.DataFrame(
            {
                "from_state": np.repeat(np.arange(k), k).tolist(),
                "to_state": np.tile(np.arange(k), k).tolist(),
                "probability": np.asarray(matrix, dtype=np.float64).flatten().tolist(),
            }
        )
        path = base / "transition_matrix.parquet"
        tm.write_parquet(path, compression=self.compression)  # type: ignore[arg-type]
        return path

    def read_states(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        model_name: str,
        version: str | None = None,
    ) -> pl.DataFrame:
        base = (
            self.root
            / f"exchange={exchange.lower()}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"model={model_name}"
        )
        if version:
            files = [base / f"version={version}" / "states.parquet"]
        else:
            files = sorted(base.rglob("states.parquet"))
        existing = [f for f in files if f.exists()]
        if not existing:
            return pl.DataFrame()
        return pl.concat([pl.read_parquet(f) for f in existing], how="diagonal_relaxed")

    def stats(self) -> dict[str, Any]:
        files = list(self.root.rglob("*.parquet"))
        return {
            "file_count": len(files),
            "total_bytes": sum(f.stat().st_size for f in files),
            "duckdb_path": str(self.duckdb_path),
        }

    def _partition(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        model_name: str,
        version: str,
    ) -> Path:
        return (
            self.root
            / f"exchange={exchange.lower()}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"model={model_name}"
            / f"version={version}"
        )

    def _register_duckdb(self, partition_dir: Path) -> None:
        states = partition_dir / "states.parquet"
        if not states.exists():
            return
        con = duckdb.connect(str(self.duckdb_path))
        try:
            path = str(states).replace("'", "''")
            con.execute(
                "CREATE OR REPLACE VIEW ss_states AS " f"SELECT * FROM read_parquet('{path}')"
            )
            tm = partition_dir / "transition_matrix.parquet"
            if tm.exists():
                tmp = str(tm).replace("'", "''")
                con.execute(
                    "CREATE OR REPLACE VIEW ss_transitions AS "
                    f"SELECT * FROM read_parquet('{tmp}')"
                )
            proba = partition_dir / "probabilities.parquet"
            if proba.exists():
                pp = str(proba).replace("'", "''")
                con.execute(
                    "CREATE OR REPLACE VIEW ss_probabilities AS "
                    f"SELECT * FROM read_parquet('{pp}')"
                )
        finally:
            con.close()
