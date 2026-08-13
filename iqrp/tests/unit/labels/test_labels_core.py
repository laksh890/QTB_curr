"""Unit tests for the Label Engineering Platform."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.core.exceptions import ConfigurationError, DataError, ValidationError
from iqrp.app.labels import (
    Label,
    LabelMeta,
    LabelPipeline,
    LabelQueryService,
    LabelSettings,
    LabelStore,
    LabelValidator,
    LabelVisualizer,
    describe_label,
    ensure_labels_loaded,
    get_registry,
    list_labels,
    meta_label_frame,
    next_n_period_return,
    probability_of_atr_move,
    probability_of_move,
    register_custom_label,
    secondary_confirmation,
    triple_barrier_frame,
)
from iqrp.app.labels.barrier import compute_triple_barrier
from iqrp.app.labels.base.registry import LabelRegistry


def _ohlcv(n: int = 160, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rets = rng.normal(0.0005, 0.01, size=n)
    close = 100 * np.cumprod(1 + rets)
    rows = []
    for i in range(n):
        c = float(close[i])
        rows.append(
            {
                "open_time": start + timedelta(hours=i),
                "open": c * (1 - 0.001),
                "high": c * (1 + 0.01),
                "low": c * (1 - 0.01),
                "close": c,
                "volume": float(10 + (i % 7)),
            }
        )
    return pl.DataFrame(rows)


@pytest.mark.unit
def test_settings_and_registry() -> None:
    settings = LabelSettings.from_hydra(overrides=["defaults.horizon=24"])
    assert settings.defaults.horizon == 24
    ensure_labels_loaded()
    names = list_labels()
    assert "future_return" in names
    assert "triple_barrier" in names
    assert "meta_label" in names
    assert "return_12" in names
    assert describe_label("future_return")["prediction_horizon"] >= 1
    assert get_registry().all_meta()


@pytest.mark.unit
def test_pipeline_all_categories() -> None:
    frame = _ohlcv(180)
    sample = [
        "future_return",
        "future_log_return",
        "future_volatility",
        "future_atr",
        "future_drawdown",
        "future_mfe",
        "future_mae",
        "future_vwap_deviation",
        "future_spread",
        "future_liquidity",
        "binary_up",
        "binary_down",
        "return_bucket",
        "volatility_bucket",
        "trend_bucket",
        "regime_class",
        "market_stress_class",
        "time_to_upper_barrier",
        "time_to_lower_barrier",
        "future_realized_volatility",
        "future_parkinson",
        "future_garman_klass",
        "future_yang_zhang",
        "future_ewma_volatility",
        "bull_bear_sideways",
        "volatility_regime",
        "liquidity_regime",
        "trend_regime",
        "bull",
        "bear",
        "sideways",
        "triple_barrier",
        "meta_label",
        "probability_label",
        "trade_filter_label",
        "return_12",
        "prob_plus_2pct",
        "prob_3atr_move",
    ]
    out, bench = LabelPipeline(max_workers=2).compute(frame, sample, parallel=True)
    assert out.height == frame.height
    assert bench.total_time_ms > 0
    for name in sample:
        cols = get_registry().describe(name).output_columns
        assert any(c in out.columns for c in cols), name


@pytest.mark.unit
def test_triple_barrier_modes() -> None:
    frame = _ohlcv(100)
    for mode in ("fixed", "atr", "volatility"):
        tb = triple_barrier_frame(frame, barrier_mode=mode, horizon=15)
        assert "tb_hit_type" in tb.columns
        assert tb.filter(pl.col("tb_hit_type").is_not_null()).height > 0
    close = frame["close"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    upper = close * 1.02
    lower = close * 0.98
    res = compute_triple_barrier(close, high, low, upper=upper, lower=lower, horizon=10)
    assert np.isfinite(res.hit_type).sum() > 0


@pytest.mark.unit
def test_meta_and_custom() -> None:
    frame = _ohlcv(80).with_columns(
        pl.Series("primary_signal", np.sign(np.random.default_rng(0).normal(size=80))),
        pl.lit(1.0).alias("confirm"),
    )
    meta = meta_label_frame(frame, primary_signal_column="primary_signal", horizon=5)
    assert "meta_label" in meta.columns
    conf = secondary_confirmation(
        frame, primary_signal_column="primary_signal", confirmation_column="confirm"
    )
    assert "secondary_confirmation" in conf.columns
    assert "return_5" in next_n_period_return(frame, 5, name="return_5").columns
    assert "prob_move" in probability_of_move(frame, threshold=0.01, horizon=5).columns
    assert "prob_atr_move" in probability_of_atr_move(frame, horizon=5).columns

    def _custom(f: pl.DataFrame) -> pl.DataFrame:
        return f.select("open_time", (pl.col("close") * 0 + 1.0).alias("custom_one"))

    register_custom_label(
        "custom_one",
        compute_fn=_custom,
        description="constant one",
        prediction_horizon=0,
        output_columns=("custom_one",),
    )
    assert "custom_one" in list_labels(category="custom")


@pytest.mark.unit
def test_validation_visualization_store(tmp_path: Path) -> None:
    frame = _ohlcv(120)
    out, _ = LabelPipeline(max_workers=1).compute(
        frame, ["future_return", "binary_up", "triple_barrier", "bull_bear_sideways"]
    )
    report = LabelValidator().validate(out)
    assert report.quality
    assert report.to_dict()["quality"]

    viz = LabelVisualizer(LabelSettings.default())
    paths = viz.write_all(
        tmp_path / "charts",
        out,
        label_columns=["binary_up", "tb_hit_type", "bull_bear_sideways", "future_return"],
    )
    assert paths["class_distribution"].exists()

    store = LabelStore(tmp_path / "labels")
    assert (
        store.write(pl.DataFrame(), exchange="binance", symbol="X", timeframe="1h", label_name="y")
        == []
    )
    with pytest.raises(DataError):
        store.write(
            pl.DataFrame({"close": [1.0]}),
            exchange="binance",
            symbol="X",
            timeframe="1h",
            label_name="y",
        )
    written = store.write(
        out.select("open_time", "future_return"),
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1h",
        label_name="future_return",
        version="1.0.0",
    )
    assert written
    store.write(
        out.select("open_time", "future_return"),
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1h",
        label_name="future_return",
        version="1.0.0",
    )
    loaded = store.read(
        exchange="binance", symbol="BTCUSDT", timeframe="1h", label_name="future_return"
    )
    assert loaded.height >= 1
    assert (
        store.update_incremental(
            out.select("open_time", "future_return"),
            exchange="binance",
            symbol="BTCUSDT",
            timeframe="1h",
            label_name="future_return",
        )
        == []
    )
    assert store.stats()["file_count"] >= 1


@pytest.mark.unit
def test_query_service(tmp_path: Path) -> None:
    frame = _ohlcv(100)
    settings = LabelSettings.from_hydra(
        overrides=[
            f"store_dir={tmp_path / 'store'}",
            f"output_dir={tmp_path / 'reports'}",
            "n_jobs=2",
        ]
    )
    svc = LabelQueryService(settings=settings, store_root=tmp_path / "store")
    result, bench = svc.compute_and_store(
        frame,
        ["future_return", "binary_up", "triple_barrier"],
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1h",
    )
    assert result.height == frame.height
    assert bench.total_time_ms > 0
    one = svc.get_label("future_return", exchange="binance", symbol="BTCUSDT", timeframe="1h")
    assert one.height >= 1
    many = svc.get_labels(
        ["future_return", "binary_up"],
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1h",
    )
    assert many.height >= 1
    assert (tmp_path / "reports" / "label_quality_report.json").exists()


@pytest.mark.unit
def test_registry_and_pipeline_errors() -> None:
    reg = LabelRegistry()

    class Bad:
        pass

    with pytest.raises(ConfigurationError):
        reg.register(Bad)  # type: ignore[arg-type]

    class Boom(Label):
        meta = LabelMeta(
            name="boom_label",
            version="1",
            description="x",
            category="test",
            prediction_horizon=1,
            required_inputs=("close",),
            output_columns=("boom_label",),
        )

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            raise RuntimeError("nope")

    class MissingOut(Label):
        meta = LabelMeta(
            name="missing_out",
            version="1",
            description="x",
            category="test",
            prediction_horizon=1,
            required_inputs=("close",),
            output_columns=("missing_out",),
        )

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            return frame.select("open_time")

    reg.register(Boom)
    reg.register(MissingOut)
    pipe = LabelPipeline(registry=reg, max_workers=1)
    frame = _ohlcv(20)
    with pytest.raises(DataError):
        pipe.compute(frame, ["boom_label"])
    with pytest.raises(ValidationError):
        MissingOut().run(frame.select("open_time"))
    with pytest.raises(ValidationError):
        MissingOut().run(frame)

    class A(Label):
        meta = LabelMeta(
            name="cycle_a",
            version="1",
            description="a",
            category="test",
            prediction_horizon=0,
            required_inputs=("close",),
            output_columns=("cycle_a",),
            dependencies=("cycle_b",),
        )

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            return frame.select("open_time", pl.lit(1.0).alias("cycle_a"))

    class B(Label):
        meta = LabelMeta(
            name="cycle_b",
            version="1",
            description="b",
            category="test",
            prediction_horizon=0,
            required_inputs=("close",),
            output_columns=("cycle_b",),
            dependencies=("cycle_a",),
        )

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            return frame.select("open_time", pl.lit(1.0).alias("cycle_b"))

    reg2 = LabelRegistry()
    reg2.register(A)
    reg2.register(B)
    with pytest.raises(ConfigurationError):
        LabelPipeline(registry=reg2).resolve_order(["cycle_a"])
