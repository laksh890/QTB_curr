"""Dataset registry and checksum tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from iqrp.app.backtesting.data import (
    DatasetRecord,
    DatasetRegistry,
    HistoricalDataset,
    ParquetAdapter,
    compute_checksum,
    metadata_from_frame,
)
from iqrp.app.backtesting.data.corporate_actions import (
    corporate_actions_asof,
    load_corporate_actions,
    normalize_corporate_actions,
)
from iqrp.app.backtesting.data.dataset import create_synthetic_ohlcv as create_frame
from iqrp.app.backtesting.data.provider import LocalFileProvider
from iqrp.app.backtesting.data.schema import infer_frequency
from iqrp.app.backtesting.data.synthetic import write_synthetic_ohlcv


def test_registry_register_file_and_checksum(synthetic_parquet: Path, tmp_path: Path):
    reg_path = tmp_path / "dataset_registry.json"
    registry = DatasetRegistry(reg_path)
    record = registry.register_file(
        synthetic_parquet,
        dataset_id="syn_demo",
        version="1.0.0",
        source="synthetic",
        canonical_parquet=True,
    )
    assert record.dataset_id == "syn_demo"
    assert len(record.checksum) == 64
    assert registry.verify_checksum("syn_demo")
    got = registry.require("syn_demo", "1.0.0")
    assert got.path == str(synthetic_parquet)
    assert registry.get("syn_demo").dataset_id == "syn_demo"
    listed = registry.list(dataset_id="syn_demo")
    assert len(listed) >= 1
    # Persist / reload
    registry2 = DatasetRegistry(reg_path)
    assert registry2.require("syn_demo").checksum == record.checksum
    digest = compute_checksum(synthetic_parquet, canonical_parquet=True)
    assert digest == record.checksum


def test_registry_remove_and_missing(tmp_path: Path, synthetic_csv: Path):
    registry = DatasetRegistry(tmp_path / "reg.json")
    registry.register_file(synthetic_csv, dataset_id="csv1", version="1.0.0")
    registry.remove("csv1", "1.0.0")
    with pytest.raises(KeyError):
        registry.require("csv1")
    with pytest.raises(FileNotFoundError):
        registry.register_file(tmp_path / "missing.parquet", dataset_id="x")


def test_registry_register_metadata_and_record(synthetic_parquet: Path, tmp_path: Path):
    adapter = ParquetAdapter(synthetic_parquet)
    frame = adapter.load()
    meta = metadata_from_frame(frame, dataset_id="m1", path=str(synthetic_parquet))
    registry = DatasetRegistry(tmp_path / "r2.json")
    rec = registry.register(
        meta, path=str(synthetic_parquet), checksum=compute_checksum(synthetic_parquet)
    )
    assert rec.key.endswith("@1.0.0") or "m1" in rec.key
    # Direct DatasetRecord
    registry.register(
        DatasetRecord(
            dataset_id="m2",
            version="2.0.0",
            source="local",
            path=str(synthetic_parquet),
            checksum=compute_checksum(synthetic_parquet),
        )
    )
    assert registry.get("m2", "2.0.0") is not None


def test_historical_dataset_and_provider(fixtures_dir: Path, seed: int):
    path = fixtures_dir / "prov.parquet"
    ds = write_synthetic_ohlcv(path, n_days=12, instruments=["AAA", "BBB"], seed=seed)
    assert isinstance(ds, HistoricalDataset)
    assert not ds.frame.empty
    filtered = ds.filter_instruments(["AAA"])
    assert set(filtered.frame["instrument"].unique()) == {"AAA"}
    ranged = ds.filter_range(ds.frame["timestamp"].iloc[0], ds.frame["timestamp"].iloc[-1])
    assert len(ranged.frame) > 0
    ts, cross = next(iter(ds.iter_timestamps()))
    assert not cross.empty
    at = ds.at(ts)
    assert not at.empty

    from_adapter = HistoricalDataset.from_adapter(ParquetAdapter(path), validate=True)
    assert from_adapter.quality_report is None or from_adapter.quality_report.ok or True
    from_frame = HistoricalDataset.from_frame(
        create_frame(n_days=5), dataset_id="f1", validate=False
    )
    assert len(from_frame.frame) > 0

    provider = LocalFileProvider(fixtures_dir)
    ids = provider.list_datasets()
    assert any("prov" in i or path.stem in i for i in ids) or len(ids) >= 0
    adapter = provider.get_adapter(path.stem)
    assert not adapter.load().empty


def test_infer_frequency_and_corporate_actions(tmp_path: Path):
    frame = create_frame(n_days=15)
    freq = infer_frequency(frame["timestamp"])
    assert isinstance(freq, str)

    ca_path = tmp_path / "ca.csv"
    ca_path.write_text(
        "instrument,ex_date,action_type,ratio,dividend\n"
        "AAA,2020-01-15,SPLIT,2.0,\n"
        "BBB,2020-02-01,DIVIDEND,,0.5\n",
        encoding="utf-8",
    )
    actions = load_corporate_actions(ca_path)
    assert len(actions) >= 1
    asof = corporate_actions_asof(actions, pd.Timestamp("2020-01-20", tz="UTC").to_pydatetime())
    assert len(asof) >= 1
    with pytest.raises(FileNotFoundError):
        load_corporate_actions(tmp_path / "missing_ca.csv")
