"""Unit tests for feature registry, pipeline, transforms, validation, store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from iqrp.app.core.exceptions import ConfigurationError
from iqrp.app.features import (
    FeaturePipeline,
    FeatureQueryService,
    FeatureStore,
    FeatureValidator,
    describe_feature,
    ensure_features_loaded,
    feature_dependencies,
    get_registry,
    list_features,
)
from iqrp.app.features.base.cache import FeatureCache
from iqrp.app.features.base.feature import Feature, FeatureMeta
from iqrp.app.features.base.registry import FeatureRegistry
from iqrp.app.features.transforms import (
    box_cox_transform,
    difference,
    expanding_window,
    lag,
    log_transform,
    normalize_minmax,
    percentage_change,
    rolling_window,
    standardize,
    winsorize,
)


def _ohlcv(n: int = 120) -> pl.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    price = 100.0
    for i in range(n):
        price += (i % 7) - 3
        rows.append(
            {
                "open_time": start + timedelta(minutes=i),
                "open": price,
                "high": price + 1.5,
                "low": price - 1.5,
                "close": price + 0.2,
                "volume": 10.0 + (i % 5),
            }
        )
    return pl.DataFrame(rows)


@pytest.mark.unit
def test_registry_lists_categories() -> None:
    ensure_features_loaded()
    names = list_features()
    assert "log_return" in names
    assert "rsi" in names
    assert "atr" in names
    assert "funding_rate" in names
    trend = list_features(category="trend")
    assert all(describe_feature(n)["category"] == "trend" for n in trend)
    deps = feature_dependencies("rolling_trend")
    assert "sma_slope" in deps


@pytest.mark.unit
def test_pipeline_compute_and_cache(tmp_path: Path) -> None:
    frame = _ohlcv(80)
    cache = FeatureCache(directory=tmp_path / "cache")
    pipe = FeaturePipeline(cache=cache, max_workers=2, use_cache=True)
    out1, _b1 = pipe.compute(frame, ["log_return", "rsi", "atr", "vwap"], parallel=True)
    assert "log_return" in out1.columns
    assert "rsi" in out1.columns
    assert "atr" in out1.columns
    _out2, b2 = pipe.compute(frame, ["log_return", "rsi", "atr", "vwap"], parallel=True)
    assert b2.cache_hit_rate >= 0.0
    assert cache.snapshot_stats()["hits"] >= 1


@pytest.mark.unit
def test_pipeline_lazy_and_incremental() -> None:
    frame = _ohlcv(60)
    pipe = FeaturePipeline(lazy=True)
    out, _ = pipe.compute(frame, ["log_return"])
    assert out.columns == frame.columns

    pipe2 = FeaturePipeline(use_cache=False, max_workers=1)
    since = frame["open_time"][40]
    out2, bench = pipe2.compute(frame, ["log_return", "momentum"], since=since, parallel=False)
    assert out2["open_time"].min() >= since
    assert bench.total_time_ms >= 0.0


@pytest.mark.unit
def test_dependency_cycle_detection() -> None:
    reg = FeatureRegistry()

    class A(Feature):
        meta = FeatureMeta(
            name="cycle_a",
            version="1",
            description="a",
            category="test",
            dependencies=("cycle_b",),
            required_columns=("close",),
            output_columns=("cycle_a",),
        )

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            return frame.select("open_time", pl.lit(1.0).alias("cycle_a"))

    class B(Feature):
        meta = FeatureMeta(
            name="cycle_b",
            version="1",
            description="b",
            category="test",
            dependencies=("cycle_a",),
            required_columns=("close",),
            output_columns=("cycle_b",),
        )

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            return frame.select("open_time", pl.lit(1.0).alias("cycle_b"))

    reg.register(A)
    reg.register(B)
    pipe = FeaturePipeline(registry=reg, use_cache=False)
    with pytest.raises(ConfigurationError):
        pipe.resolve_order(["cycle_a"])


@pytest.mark.unit
def test_transforms_and_boxcox() -> None:
    frame = _ohlcv(50).select("close", "volume")
    out = lag(frame, ["close"], 1)
    out = difference(out, ["close"], 1)
    out = percentage_change(out, ["close"], 1)
    out = rolling_window(out, ["close"], 5, agg="mean")
    out = expanding_window(out, ["close"], agg="mean")
    out = normalize_minmax(out, ["close"])
    out = standardize(out, ["close"])
    out = winsorize(out, ["close"])
    out = log_transform(out, ["volume"])
    out = box_cox_transform(out, "volume")
    assert "close_lag1" in out.columns
    assert "volume_boxcox" in out.columns


@pytest.mark.unit
def test_feature_validator() -> None:
    frame = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, None],
            "b": [1.0, 2.0, 3.0, 4.0],
            "c": [1.0, 1.0, 1.0, 1.0],
            "d": [1.0, 2.0, 3.0, 4.0],
        }
    )
    report = FeatureValidator(corr_threshold=0.99).validate(frame)
    assert report.nan_counts["a"] == 1
    assert "c" in report.constant_features
    assert "nan_counts" in report.to_dict()


@pytest.mark.unit
def test_feature_store_and_query(tmp_path: Path) -> None:
    frame = _ohlcv(90)
    pipe = FeaturePipeline(use_cache=False, max_workers=2)
    enriched, _ = pipe.compute(
        frame, ["log_return", "rsi", "atr", "hour", "relative_volume"], parallel=True
    )
    store = FeatureStore(tmp_path / "features")
    paths = store.write(
        enriched,
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        feature_group="mixed",
    )
    assert paths
    loaded = store.read(exchange="binance", symbol="BTCUSDT", timeframe="1m", feature_group="mixed")
    assert loaded.height == enriched.height
    assert (
        store.update_incremental(
            enriched,
            exchange="binance",
            symbol="BTCUSDT",
            timeframe="1m",
            feature_group="mixed",
        )
        == []
    )

    svc = FeatureQueryService(store=store, pipeline=pipe)
    many = svc.get_features(
        ["log_return", "rsi"], exchange="binance", symbol="BTCUSDT", timeframe="1m"
    )
    assert many.height >= 1
    assert svc.describe_feature("atr")["name"] == "atr"
    assert store.stats()["file_count"] >= 1


@pytest.mark.unit
def test_all_categories_smoke() -> None:
    ensure_features_loaded()
    reg = get_registry()
    frame = _ohlcv(100)
    sample = [
        "log_return",
        "momentum",
        "rolling_std",
        "vwap",
        "bid_ask_spread",
        "trade_imbalance",
        "basis",
        "zscore",
        "hour",
        "beta_to_benchmark",
    ]
    pipe = FeaturePipeline(use_cache=False, max_workers=2)
    out, bench = pipe.compute(frame, sample, parallel=True)
    for name in sample:
        cols = reg.describe(name).output_columns
        assert any(c in out.columns for c in cols)
    assert bench.memory_bytes_estimate >= 0
