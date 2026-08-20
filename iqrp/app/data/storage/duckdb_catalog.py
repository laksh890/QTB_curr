"""DuckDB catalog that auto-registers Parquet partitions for SQL access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from loguru import logger

from iqrp.app.core.exceptions import DataError
from iqrp.app.data.storage.parquet_store import ParquetStore
from iqrp.app.data.types import MarketDataType


class DuckDBCatalog:
    """Register parquet datasets as DuckDB views and run SQL queries."""

    def __init__(self, database_path: Path, store: ParquetStore) -> None:
        self.database_path = Path(database_path)
        self.store = store
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.database_path))

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DuckDBCatalog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def register_all(self) -> list[str]:
        """Create or replace views for every known market-data type present on disk."""
        registered: list[str] = []
        for data_type in MarketDataType:
            view = self.register_data_type(data_type)
            if view:
                registered.append(view)
        return registered

    def register_data_type(self, data_type: MarketDataType | str) -> str | None:
        files = self.store.list_parquet_files(data_type)
        if not files:
            return None
        view_name = f"iqrp_{str(data_type).replace('/', '_')}"
        if not view_name.replace("_", "").isalnum():
            raise DataError("Invalid DuckDB view name", code="DUCKDB_BAD_VIEW")
        # Use hive partitioning discovery via glob.
        glob_path = str(self.store.root / str(data_type) / "**" / "*.parquet").replace("'", "''")
        sql = (
            f"CREATE OR REPLACE VIEW {view_name} AS "
            f"SELECT * FROM read_parquet('{glob_path}', hive_partitioning=true, union_by_name=true)"
        )
        try:
            self._conn.execute(sql)
        except Exception as exc:
            raise DataError(
                f"Failed to register DuckDB view '{view_name}': {exc}",
                code="DUCKDB_REGISTER_FAILED",
                details={"glob": glob_path},
            ) from exc
        logger.info("duckdb_register view={} files={}", view_name, len(files))
        return view_name

    def register_files(self, files: list[Path], *, view_name: str) -> str:
        if not files:
            raise DataError("No parquet files to register", code="DUCKDB_NO_FILES")
        if not view_name.replace("_", "").isalnum():
            raise DataError("Invalid DuckDB view name", code="DUCKDB_BAD_VIEW")
        file_list = ", ".join(f"'{f.as_posix().replace(chr(39), chr(39)+chr(39))}'" for f in files)
        sql = (
            f"CREATE OR REPLACE VIEW {view_name} AS "
            f"SELECT * FROM read_parquet([{file_list}], union_by_name=true)"
        )
        self._conn.execute(sql)
        logger.info("duckdb_register_files view={} count={}", view_name, len(files))
        return view_name

    def sql(self, query: str, params: list[Any] | None = None) -> pl.DataFrame:
        """Execute SQL and return a Polars DataFrame."""
        try:
            result = self._conn.execute(query, params or []).pl()
        except Exception as exc:
            raise DataError(
                f"DuckDB query failed: {exc}",
                code="DUCKDB_QUERY_FAILED",
                details={"query": query[:500]},
            ) from exc
        return result

    def table_names(self) -> list[str]:
        rows = self._conn.execute("SHOW TABLES").fetchall()
        return [str(r[0]) for r in rows]
