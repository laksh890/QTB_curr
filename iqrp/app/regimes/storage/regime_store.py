"""Persist regime detection artifacts to Parquet + DuckDB."""

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

from iqrp.app.regimes.base.regime import RegimeResult


class RegimeStore:
    """Hive-style parquet store with optional DuckDB registration."""

    def __init__(
        self,
        root: Path,
        *,
        duckdb_path: Path | None = None,
        compression: str = "zstd",
        register_duckdb: bool = True,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.register_duckdb = register_duckdb
        self.duckdb_path = Path(duckdb_path) if duckdb_path else self.root / "regimes.duckdb"
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    def write_result(
        self,
        result: RegimeResult,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        timestamp_column: str = "open_time",
    ) -> dict[str, Path]:
        base = (
            self.root
            / f"exchange={exchange.lower()}"
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"model={result.model_name}"
            / f"version={result.model_version}"
        )
        base.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        states = result.to_frame(timestamp_column=timestamp_column)
        states_path = base / "states.parquet"
        states.write_parquet(states_path, compression=self.compression)  # type: ignore[arg-type]
        paths["states"] = states_path

        k = int(result.transition_matrix.shape[0])
        from_state = np.repeat(np.arange(k), k)
        to_state = np.tile(np.arange(k), k)
        tm = pl.DataFrame(
            {
                "from_state": from_state.tolist(),
                "to_state": to_state.tolist(),
                "probability": result.transition_matrix.flatten().tolist(),
            }
        )
        tm_path = base / "transition_matrix.parquet"
        tm.write_parquet(tm_path, compression=self.compression)  # type: ignore[arg-type]
        paths["transition_matrix"] = tm_path

        if result.state_probabilities.ndim == 2:
            proba_data: dict[str, Any] = {}
            if timestamp_column in states.columns:
                proba_data[timestamp_column] = states[timestamp_column].to_list()
            for j in range(result.state_probabilities.shape[1]):
                proba_data[f"proba_{j}"] = result.state_probabilities[:, j].tolist()
            proba_path = base / "probabilities.parquet"
            pl.DataFrame(proba_data).write_parquet(
                proba_path, compression=self.compression  # type: ignore[arg-type]
            )
            paths["probabilities"] = proba_path

        if result.forecast is not None:
            fc_path = base / "forecast.json"
            fc_path.write_text(json.dumps(result.forecast.to_dict(), indent=2), encoding="utf-8")
            paths["forecast"] = fc_path

        meta_path = base / "metadata.json"
        meta_path.write_text(
            json.dumps(
                {
                    "model_name": result.model_name,
                    "model_version": result.model_version,
                    "feature_columns": list(result.feature_columns),
                    "metadata": result.metadata,
                    "written_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        paths["metadata"] = meta_path

        if self.register_duckdb:
            self._register_duckdb(base)

        logger.info("regime_store_write base={} files={}", base, list(paths))
        return paths

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

    def _register_duckdb(self, partition_dir: Path) -> None:
        states = partition_dir / "states.parquet"
        if not states.exists():
            return
        con = duckdb.connect(str(self.duckdb_path))
        try:
            path = str(states).replace("'", "''")
            # Local store paths only — not end-user SQL.
            con.execute(
                "CREATE OR REPLACE VIEW regime_states AS " f"SELECT * FROM read_parquet('{path}')"
            )
            tm = partition_dir / "transition_matrix.parquet"
            if tm.exists():
                tmp = str(tm).replace("'", "''")
                con.execute(
                    "CREATE OR REPLACE VIEW regime_transitions AS "
                    f"SELECT * FROM read_parquet('{tmp}')"
                )
        finally:
            con.close()

    def stats(self) -> dict[str, Any]:
        files = list(self.root.rglob("*.parquet"))
        return {
            "file_count": len(files),
            "total_bytes": sum(f.stat().st_size for f in files),
            "duckdb_path": str(self.duckdb_path),
        }
