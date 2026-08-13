"""Integration + property tests for the feature platform."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given, settings, strategies as st

from iqrp.app.features import FeaturePipeline, FeatureQueryService, list_features
from iqrp.app.features.base.registry import ensure_features_loaded


def _frame(n: int) -> pl.DataFrame:
    start = datetime(2024, 6, 1, tzinfo=UTC)
    price = 50.0
    rows = []
    for i in range(n):
        price = max(1.0, price + ((i * 17) % 5) - 2)
        rows.append(
            {
                "open_time": start + timedelta(minutes=i),
                "open": price,
                "high": price + 2,
                "low": max(0.1, price - 2),
                "close": price + 0.1,
                "volume": 5 + (i % 9),
            }
        )
    return pl.DataFrame(rows)


@pytest.mark.integration
def test_end_to_end_feature_store(tmp_path: Path) -> None:
    ensure_features_loaded()
    names = [
        "log_return",
        "multi_period_return",
        "rsi",
        "macd_components",
        "atr",
        "parkinson_volatility",
        "relative_volume",
        "obv",
        "quoted_spread",
        "funding_momentum",
        "hurst_exponent",
        "weekend",
        "relative_strength_vs_benchmark",
    ]
    frame = _frame(150)
    svc = FeatureQueryService(store_root=tmp_path / "features")
    result, bench = svc.compute_and_store(
        frame,
        names,
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        feature_group="research",
    )
    assert result.height == frame.height
    assert bench.total_time_ms > 0
    stored = svc.get_features(
        ["log_return", "rsi", "atr"],
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
    )
    assert stored.height >= 1
    assert "log_return" in list_features(category="trend")


@pytest.mark.unit
@settings(max_examples=15, deadline=None)
@given(n=st.integers(min_value=40, max_value=90))
def test_property_pipeline_row_count(n: int) -> None:
    frame = _frame(n)
    pipe = FeaturePipeline(use_cache=False, max_workers=1)
    out, _ = pipe.compute(frame, ["log_return", "roc", "rolling_mean"], parallel=False)
    assert out.height == n
    assert "log_return" in out.columns
