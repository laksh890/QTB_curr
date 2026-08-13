"""Integration + synthetic market tests for labels."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.labels import LabelPipeline, LabelQueryService, LabelSettings, LabelValidator


def _synthetic_trend(n: int = 240, seed: int = 9) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2023, 1, 1, tzinfo=UTC)
    # Clear upward then downward regimes
    rets = np.concatenate([rng.normal(0.002, 0.005, n // 2), rng.normal(-0.002, 0.008, n - n // 2)])
    close = 50 * np.cumprod(1 + rets)
    rows = []
    for i in range(n):
        c = float(close[i])
        rows.append(
            {
                "open_time": start + timedelta(hours=i),
                "open": c,
                "high": c * 1.015,
                "low": c * 0.985,
                "close": c,
                "volume": float(5 + i % 4),
                "primary_signal": float(np.sign(rets[i - 1]) if i else 0.0),
            }
        )
    return pl.DataFrame(rows)


@pytest.mark.integration
def test_synthetic_end_to_end(tmp_path: Path) -> None:
    frame = _synthetic_trend()
    settings = LabelSettings.from_hydra(
        overrides=[
            f"store_dir={tmp_path / 'store'}",
            f"output_dir={tmp_path / 'out'}",
            "defaults.horizon=8",
            "triple_barrier.horizon=16",
            "n_jobs=2",
        ]
    )
    names = [
        "future_return",
        "binary_up",
        "triple_barrier",
        "meta_label",
        "bull_bear_sideways",
        "future_realized_volatility",
        "return_bucket",
    ]
    svc = LabelQueryService(settings=settings, store_root=tmp_path / "store")
    out, bench = svc.compute_and_store(
        frame,
        names,
        exchange="binance",
        symbol="ETHUSDT",
        timeframe="1h",
        write_reports=True,
    )
    assert out.height == frame.height
    assert bench.total_time_ms > 0
    assert (
        svc.get_label("future_return", exchange="binance", symbol="ETHUSDT", timeframe="1h").height
        >= 1
    )
    report = LabelValidator(settings).validate(out)
    assert report.quality
    assert (tmp_path / "out" / "charts").exists()


@pytest.mark.integration
def test_incremental_store(tmp_path: Path) -> None:
    frame = _synthetic_trend(120)
    settings = LabelSettings.from_hydra(overrides=[f"store_dir={tmp_path / 's'}"])
    svc = LabelQueryService(settings=settings, store_root=tmp_path / "s")
    svc.compute_and_store(
        frame[:80],
        ["future_return"],
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1h",
        write_reports=False,
    )
    result, _ = svc.compute_and_store(
        frame,
        ["future_return"],
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1h",
        incremental=True,
        write_reports=False,
    )
    assert result.height == frame.height
    # Parallel smoke
    out, _ = LabelPipeline(max_workers=2).compute(
        frame, ["future_return", "binary_up", "triple_barrier"], parallel=True
    )
    assert "tb_return" in out.columns
