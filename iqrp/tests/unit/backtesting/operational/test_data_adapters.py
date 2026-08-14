"""Data adapter tests: CSV, Parquet, Arrow/Feather."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from iqrp.app.backtesting.data import (
    CSVAdapter,
    DatasetValidator,
    ParquetAdapter,
    file_sha256,
    parquet_canonical_sha256,
)
from iqrp.app.backtesting.data.synthetic import generate_synthetic_ohlcv, write_synthetic_ohlcv


def test_csv_adapter_load_and_validate(synthetic_csv: Path):
    adapter = CSVAdapter(synthetic_csv, dataset_id="csv_demo")
    frame = adapter.load()
    assert not frame.empty
    assert {"timestamp", "instrument", "open", "high", "low", "close", "volume"}.issubset(
        frame.columns
    )
    report = adapter.validate(raise_on_critical=True)
    assert report.ok
    assert "AAA" in adapter.available_instruments()
    dates = adapter.available_dates()
    assert len(dates) > 0
    meta = adapter.metadata()
    assert meta.dataset_id == "csv_demo"
    subset = adapter.load_instrument("AAA")
    assert (subset["instrument"] == "AAA").all()
    uni = adapter.load_universe(["AAA"])
    assert set(uni["instrument"].unique()) == {"AAA"}
    ranged = adapter.load_range(dates[0], dates[-1])
    assert len(ranged) >= 1
    adapter.clear_cache()
    assert adapter.load(refresh=True) is not None


def test_csv_adapter_directory(tmp_path: Path):
    d = tmp_path / "csv_dir"
    d.mkdir()
    write_synthetic_ohlcv(d / "a.csv", n_days=5, instruments=["AAA"], seed=1)
    write_synthetic_ohlcv(d / "b.csv", n_days=5, instruments=["BBB"], seed=2)
    adapter = CSVAdapter(d)
    frame = adapter.load()
    assert set(frame["instrument"].unique()) >= {"AAA", "BBB"}


def test_csv_adapter_missing_and_empty_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CSVAdapter(tmp_path / "missing.csv")
    empty = tmp_path / "empty_csv"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        CSVAdapter(empty).load()


def test_parquet_adapter_file_and_checksum(synthetic_parquet: Path):
    adapter = ParquetAdapter(synthetic_parquet)
    frame = adapter.load()
    assert len(frame) > 0
    digest = adapter.checksum()
    assert len(digest) == 64
    assert digest == file_sha256(synthetic_parquet)
    canonical = adapter.checksum(canonical=True)
    assert len(canonical) == 64
    assert parquet_canonical_sha256(synthetic_parquet) == canonical


def test_parquet_adapter_feather_and_arrow(synthetic_feather: Path, fixtures_dir: Path):
    feather_adapter = ParquetAdapter(synthetic_feather, source="feather")
    assert not feather_adapter.load().empty

    arrow_path = fixtures_dir / "synthetic_bars.arrow"
    write_synthetic_ohlcv(arrow_path, n_days=10, instruments=["AAA"], seed=3)
    arrow_adapter = ParquetAdapter(arrow_path)
    assert not arrow_adapter.load().empty


def test_parquet_adapter_in_memory_table():
    frame = generate_synthetic_ohlcv(n_days=8, instruments=["X"], seed=9)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    adapter = ParquetAdapter(table, dataset_id="mem")
    loaded = adapter.load()
    assert len(loaded) == len(frame)
    assert len(adapter.checksum()) == 64


def test_parquet_adapter_directory_hive(tmp_path: Path):
    d = tmp_path / "pq_dir"
    d.mkdir()
    frame = generate_synthetic_ohlcv(n_days=6, instruments=["AAA", "BBB"], seed=4)
    frame.to_parquet(d / "part-0.parquet", index=False)
    adapter = ParquetAdapter(d)
    assert len(adapter.load()) == len(frame)


def test_parquet_adapter_columns_filter(synthetic_parquet: Path):
    adapter = ParquetAdapter(
        synthetic_parquet,
        columns=["timestamp", "instrument", "open", "high", "low", "close", "volume"],
    )
    frame = adapter.load()
    assert "close" in frame.columns


def test_parquet_adapter_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ParquetAdapter(tmp_path / "nope.parquet")


def test_adapter_with_custom_validator(synthetic_parquet: Path):
    validator = DatasetValidator(fail_on_duplicates=True)
    adapter = ParquetAdapter(synthetic_parquet, validator=validator, normalize=True)
    report = adapter.validate()
    assert report.ok
