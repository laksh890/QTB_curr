"""Broad coverage tests for the feature engineering platform."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from iqrp.app.core.exceptions import ConfigurationError, DataError, ValidationError
from iqrp.app.features import (
    Feature,
    FeatureCache,
    FeatureMeta,
    FeaturePipeline,
    FeatureQueryService,
    FeatureRegistry,
    FeatureStore,
    FeatureValidator,
    ensure_features_loaded,
    get_feature,
    get_features,
    get_metadata,
    get_registry,
    list_features,
    list_metadata,
)
from iqrp.app.features._polars_utils import require_ohlcv
from iqrp.app.features.base.pipeline import PipelineBenchmarks
from iqrp.app.features.base.registry import feature_factory
from iqrp.app.features.statistical.features import _rolling_acf, _rolling_hurst
from iqrp.app.features.transforms import (
    box_cox_transform,
    expanding_window,
    rolling_window,
)


def _market(n: int = 120, *, with_aux: bool = False) -> pl.DataFrame:
    start = datetime(2024, 1, 6, tzinfo=UTC)  # Saturday for weekend/holiday paths
    rows = []
    price = 100.0
    for i in range(n):
        price = max(1.0, price + ((i * 3) % 7) - 3)
        row: dict[str, object] = {
            "open_time": start + timedelta(minutes=i),
            "open": price,
            "high": price + 1.5,
            "low": price - 1.5,
            "close": price + 0.25,
            "volume": 10.0 + (i % 5),
        }
        if with_aux:
            row.update(
                {
                    "best_bid": price - 0.1,
                    "best_ask": price + 0.1,
                    "bid_size": 5.0 + i % 3,
                    "ask_size": 4.0 + i % 2,
                    "funding_rate": 0.0001 * ((i % 5) - 2),
                    "open_interest": 1000.0 + i,
                    "long_short_ratio": 1.0 + 0.01 * (i % 4),
                    "liquidation_count": float(i % 3),
                    "liquidation_volume": float(i % 7),
                    "mark_price": price + 0.05,
                    "index_price": price,
                    "benchmark_close": 50.0 + 0.1 * i,
                }
            )
        rows.append(row)
    return pl.DataFrame(rows)


@pytest.mark.unit
def test_compute_all_registered_features() -> None:
    ensure_features_loaded()
    names = list_features()
    assert len(names) >= 80
    frame = _market(160, with_aux=True)
    pipe = FeaturePipeline(use_cache=False, max_workers=4)
    out, bench = pipe.compute(frame, names, parallel=True)
    assert out.height == frame.height
    assert bench.total_time_ms > 0
    assert PipelineBenchmarks().to_dict()["feature_times_ms"] == {}
    reg = get_registry()
    for name in names:
        cols = reg.describe(name).output_columns
        assert any(c in out.columns for c in cols), name


@pytest.mark.unit
def test_registry_edge_cases() -> None:
    ensure_features_loaded()
    reg = get_registry()
    assert reg.all_meta()
    factory = feature_factory("log_return")
    assert factory().meta.name == "log_return"

    local = FeatureRegistry()

    class Bad:
        pass

    with pytest.raises(ConfigurationError):
        local.register(Bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        local.get_class("missing")

    class Ok(Feature):
        meta = FeatureMeta(
            name="tmp_ok",
            version="1",
            description="ok",
            category="test",
            required_columns=("close",),
            output_columns=("tmp_ok",),
        )

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            return frame.select("open_time", pl.lit(1.0).alias("tmp_ok"))

    local.register(Ok)
    assert local.get("tmp_ok").meta.name == "tmp_ok"
    local.clear()
    with pytest.raises(ConfigurationError):
        local.get("tmp_ok")

    ensure_features_loaded(modules=["iqrp.app.features.trend"])
    meta = get_metadata("log_return")
    assert meta["name"] == "log_return"
    assert list_metadata(category="trend")


@pytest.mark.unit
def test_feature_run_validation_errors() -> None:
    class BrokenOut(Feature):
        meta = FeatureMeta(
            name="broken_out",
            version="1",
            description="x",
            category="test",
            required_columns=("close",),
            output_columns=("missing_col",),
        )

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            return frame.select("open_time", pl.lit(1.0).alias("other"))

    frame = _market(10)
    with pytest.raises(ValidationError):
        BrokenOut().run(frame.select("open_time"))
    with pytest.raises(ValidationError):
        BrokenOut().run(frame)
    with pytest.raises(ValidationError):
        require_ohlcv(pl.DataFrame({"close": [1.0]}))


@pytest.mark.unit
def test_cache_disk_eviction_and_clear(tmp_path: Path) -> None:
    cache = FeatureCache(directory=tmp_path / "c", max_entries=2)
    frame = pl.DataFrame({"open_time": [datetime(2024, 1, 1, tzinfo=UTC)], "close": [1.0]})
    k1 = FeatureCache.make_key("a", "1", {}, frame, columns=("close",))
    k2 = FeatureCache.make_key("b", "1", {}, frame, columns=("close",))
    k3 = FeatureCache.make_key("c", "1", {}, frame, columns=("close",))
    cache.put(k1, frame)
    cache.put(k2, frame)
    cache.put(k3, frame)  # evicts oldest
    # disk reload path
    cold = FeatureCache(directory=tmp_path / "c", max_entries=10)
    assert cold.get(k3) is not None
    cold.clear()
    assert cold.snapshot_stats()["entries"] == 0
    assert cold.get("missing") is None


@pytest.mark.unit
def test_pipeline_failure_and_edge_paths() -> None:
    reg = FeatureRegistry()

    class Boom(Feature):
        meta = FeatureMeta(
            name="boom",
            version="1",
            description="boom",
            category="test",
            required_columns=("close",),
            output_columns=("boom",),
        )

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            raise RuntimeError("explode")

    class Boom2(Feature):
        meta = FeatureMeta(
            name="boom2",
            version="1",
            description="boom2",
            category="test",
            required_columns=("close",),
            output_columns=("boom2",),
        )

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            raise RuntimeError("explode2")

    class EmptyFeat(Feature):
        meta = FeatureMeta(
            name="empty_feat",
            version="1",
            description="empty",
            category="test",
            required_columns=("close",),
            output_columns=("empty_feat",),
        )

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            return pl.DataFrame(
                schema={"open_time": frame.schema["open_time"], "empty_feat": pl.Float64}
            )

    class NoTime(Feature):
        meta = FeatureMeta(
            name="no_time",
            version="1",
            description="no time",
            category="test",
            required_columns=("close",),
            output_columns=("no_time",),
        )

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            return pl.DataFrame({"no_time": [1.0] * frame.height})

    reg.register(Boom)
    reg.register(Boom2)
    reg.register(EmptyFeat)
    reg.register(NoTime)
    pipe = FeaturePipeline(registry=reg, use_cache=False, max_workers=2)
    frame = _market(20)
    with pytest.raises(DataError):
        pipe.compute(frame, ["boom"], parallel=False)
    with pytest.raises(DataError):
        pipe.compute(frame, ["boom", "boom2"], parallel=True)
    out, _ = pipe.compute(frame, ["empty_feat"], parallel=False)
    assert out.height == frame.height

    # join without open_time on feature side using base without duplicate cols
    out2 = FeaturePipeline._join_outputs(
        frame.select("close"),
        pl.DataFrame({"no_time": [1.0] * frame.height}),
    )
    assert "no_time" in out2.columns
    assert FeaturePipeline._join_outputs(frame, pl.DataFrame()).height == frame.height
    assert (
        FeaturePipeline._join_outputs(
            frame.select("close", pl.lit(1.0).alias("no_time")),
            pl.DataFrame({"no_time": [2.0] * frame.height}),
        ).height
        == frame.height
    )

    since = frame["open_time"][-1] + timedelta(minutes=1)
    out3, _ = pipe.compute(frame, ["empty_feat"], since=since)
    assert out3.height == frame.height


@pytest.mark.unit
def test_store_and_query_paths(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "fs")
    assert (
        store.write(
            pl.DataFrame(),
            exchange="binance",
            symbol="X",
            timeframe="1m",
            feature_group="g",
        )
        == []
    )
    with pytest.raises(DataError):
        store.write(
            pl.DataFrame({"close": [1.0]}),
            exchange="binance",
            symbol="X",
            timeframe="1m",
            feature_group="g",
        )
    assert store.read(exchange="none", symbol="X", timeframe="1m").is_empty()

    frame = _market(40)
    pipe = FeaturePipeline(use_cache=False)
    enriched, _ = pipe.compute(frame, ["log_return", "rsi", "hour"])
    store.write(
        enriched,
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        feature_group="research",
    )
    # merge path
    store.write(
        enriched,
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        feature_group="research",
    )
    start = enriched["open_time"][5]
    end = enriched["open_time"][20]
    clipped = store.read(
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        feature_group="research",
        start=start,
        end=end,
    )
    assert clipped.height >= 1

    more = _market(10)
    more = more.with_columns(pl.col("open_time") + timedelta(minutes=40))
    enriched2, _ = pipe.compute(more, ["log_return"])
    paths = store.update_incremental(
        enriched2,
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        feature_group="research",
    )
    assert paths

    svc = FeatureQueryService(store=store, pipeline=pipe)
    one = svc.get_feature(
        "log_return",
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        feature_group="research",
        start=start,
        end=end,
    )
    assert one.height >= 1
    many = svc.get_features(
        ["log_return", "rsi"],
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        feature_group="research",
    )
    assert many.height >= 1
    assert (
        get_feature(
            "log_return",
            exchange="binance",
            symbol="BTCUSDT",
            timeframe="1m",
            store_root=tmp_path / "fs",
            feature_group="research",
        ).height
        >= 1
    )
    assert (
        get_features(
            ["log_return"],
            exchange="binance",
            symbol="BTCUSDT",
            timeframe="1m",
            store_root=tmp_path / "fs",
            feature_group="research",
        ).height
        >= 1
    )

    # incremental store path via query service
    svc.compute_and_store(
        more,
        ["log_return"],
        exchange="binance",
        symbol="ETHUSDT",
        timeframe="1m",
        feature_group="all",
        incremental_since=more["open_time"][0],
    )
    assert store.stats()["file_count"] >= 1


@pytest.mark.unit
def test_transforms_branches_and_stats_helpers() -> None:
    frame = _market(30).select("close", "volume")
    out = rolling_window(frame, ["close"], 5, agg="std")
    out = rolling_window(out, ["close"], 5, agg="sum")
    out = expanding_window(out, ["close"], agg="sum")
    with pytest.raises(ValueError):
        rolling_window(frame, ["close"], 5, agg="mad")
    with pytest.raises(ValueError):
        expanding_window(frame, ["close"], agg="mad")
    tiny = pl.DataFrame({"volume": [1.0, 2.0]})
    assert "volume_boxcox" in box_cox_transform(tiny, "volume").columns

    import numpy as np

    const = np.ones(40)
    h = _rolling_hurst(const, 10)
    assert any(x == 0.5 or x != x for x in h)  # 0.5 or nan
    bad = np.array([np.nan] * 20)
    _rolling_hurst(bad, 10)
    acf = _rolling_acf(const, 10, 1)
    assert acf[-1] == 0.0
    _rolling_acf(np.arange(10, dtype=float), 5, 100)


@pytest.mark.unit
def test_validator_inf_and_low_variance() -> None:
    frame = pl.DataFrame(
        {
            "a": [1.0, 1.000000000001, 1.0, 1.0],
            "b": [1.0, float("inf"), 3.0, 4.0],
            "c": [1.0, 2.0, 3.0, 4.0],
            "d": [1.0, 2.0, 3.0, 4.0],
        }
    )
    report = FeatureValidator(variance_epsilon=1.0, corr_threshold=0.99).validate(frame)
    assert report.inf_counts["b"] >= 1
    assert report.to_dict()["highly_correlated_pairs"] is not None


@pytest.mark.unit
def test_remaining_edge_branches(tmp_path: Path) -> None:
    # cross-asset fallbacks without benchmark_close
    frame = _market(60, with_aux=False)
    pipe = FeaturePipeline(use_cache=False)
    out, _ = pipe.compute(
        frame,
        ["spread_to_benchmark", "beta_to_benchmark", "rolling_kurtosis"],
        parallel=False,
    )
    assert "spread_to_benchmark" in out.columns

    # constant window for kurtosis std==0 branch
    flat = frame.with_columns(pl.lit(10.0).alias("close"))
    out_k, _ = pipe.compute(flat, ["rolling_kurtosis"], parallel=False)
    assert "rolling_kurtosis" in out_k.columns

    # query join without open_time + empty get_features
    store = FeatureStore(tmp_path / "q")
    store.write(
        pl.DataFrame({"open_time": [datetime(2024, 1, 1, tzinfo=UTC)], "log_return": [0.1]}),
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        feature_group="x",
    )
    svc = FeatureQueryService(store=store)
    missing = svc.get_features(
        ["log_return"],
        exchange="bybit",
        symbol="NOPE",
        timeframe="1m",
        feature_group="x",
    )
    assert missing.is_empty()
    # Stored frame has open_time but not macd outputs → select open_time only
    partial = svc.get_feature(
        "macd_components",
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        feature_group="x",
    )
    assert "macd" not in partial.columns

    # store path that exists but has no parquet yet
    empty_root = tmp_path / "empty_tree"
    (empty_root / "exchange=binance" / "symbol=BTCUSDT" / "timeframe=1m" / "feature_group=g").mkdir(
        parents=True
    )
    assert (
        FeatureStore(empty_root)
        .read(exchange="binance", symbol="BTCUSDT", timeframe="1m", feature_group="g")
        .is_empty()
    )

    # non-datetime timestamp coercion
    weird = pl.DataFrame({"open_time": ["2024-01-01T00:00:00+00:00"], "v": [1.0]})
    # cast to string already — write path converts via fromisoformat
    FeatureStore(tmp_path / "weird").write(
        weird.with_columns(pl.col("open_time").str.to_datetime(time_zone="UTC")),
        exchange="binance",
        symbol="X",
        timeframe="1m",
        feature_group="g",
    )
