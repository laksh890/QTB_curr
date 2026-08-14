"""Integration tests for horizon research engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iqrp.app.backtesting.data.synthetic import generate_synthetic_ohlcv
from iqrp.app.backtesting.horizon import HorizonResearchConfig, HorizonResearchEngine
from iqrp.app.backtesting.horizon.types import HorizonStatus
from iqrp.app.backtesting.strategy.long_short_momentum import LongShortMomentumStrategy
from iqrp.app.backtesting.strategy.registry import StrategyRegistry


@pytest.fixture
def daily_frame():
    return generate_synthetic_ohlcv(
        n_days=180, freq="1d", seed=11, instruments=["DEMO"], start="2018-01-01"
    )


@pytest.fixture
def intraday_fixture():
    """Small deterministic intraday fixture for software tests only."""
    start = pd.Timestamp("2020-01-02 09:30", tz="UTC")
    ts = pd.date_range(start, periods=78 * 5, freq="5min", tz="UTC")  # ~5 sessions
    rng = np.random.default_rng(42)
    close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.002, size=len(ts)))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "instrument": "INTRA",
            "open": np.r_[close[0], close[:-1]],
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.integers(100, 5000, size=len(ts)).astype(float),
        }
    )


def test_daily_dataset_marks_intraday_unavailable(daily_frame):
    cfg = HorizonResearchConfig(
        data_timeframes=["1m", "5m", "15m", "30m", "1h", "4h", "1D"],
        holding_bars=[1, 5],
        instrument="DEMO",
        robust_gates={
            "min_trades": 1,
            "min_oos_sharpe": -10,
            "min_oos_expectancy": -1e9,
            "min_neighborhood_stability": 0.0,
            "max_drawdown": 1.0,
            "require_positive_net_expectancy": False,
        },
    )
    eng = HorizonResearchEngine(daily_frame, config=cfg)
    results = eng.sweep()
    by_tf = {}
    for r in results:
        by_tf.setdefault(str(r.spec.data_timeframe), []).append(r.status)
    for tf in ["1m", "5m", "15m", "30m", "1h", "4h"]:
        assert all(s == HorizonStatus.UNAVAILABLE for s in by_tf[tf])
    assert any(s != HorizonStatus.UNAVAILABLE for s in by_tf["1D"])
    report = eng.report()
    assert report["n_unavailable"] > 0
    assert "matrix" in report
    matrix = eng.matrix()
    assert any(row["status"] == "UNAVAILABLE" for row in matrix)
    assert any(row["data_timeframe"] == "1D" for row in matrix)


def test_multi_horizon_sweep_intraday_fixture(intraday_fixture):
    cfg = HorizonResearchConfig(
        data_timeframes=["1m", "5m", "15m", "1h"],
        signal_timeframes=["5m", "15m", "1h"],
        holding_bars=[1, 3, 5],
        instrument="INTRA",
        commission_bps=0.5,
        spread_bps=1.0,
        slippage_bps=1.0,
        robust_gates={
            "min_trades": 1,
            "min_oos_sharpe": -10,
            "min_oos_expectancy": -1e9,
            "min_neighborhood_stability": 0.0,
            "max_drawdown": 1.0,
            "require_positive_net_expectancy": False,
        },
    )
    eng = HorizonResearchEngine(intraday_fixture, config=cfg)
    results = eng.sweep()
    # 1m unavailable from 5m native
    assert any(r.status == HorizonStatus.UNAVAILABLE for r in results)
    available = [r for r in results if r.status != HorizonStatus.UNAVAILABLE]
    assert available
    assert all("net_sharpe" in r.metrics for r in available)
    assert all(r.trade_frequency for r in available)
    assert all("transaction_costs" in r.costs for r in available)
    sel = eng.selection()
    assert "best_in_sample_horizon" in sel
    assert "best_robust_horizon" in sel
    assert "are_identical" in sel


def test_long_short_and_repeated_trades(intraday_fixture):
    cfg = HorizonResearchConfig(
        data_timeframes=["5m"],
        signal_timeframes=["5m"],
        holding_bars=[1, 2],
        instrument="INTRA",
        allow_short=True,
        robust_gates={
            "min_trades": 1,
            "min_oos_sharpe": -10,
            "min_oos_expectancy": -1e9,
            "min_neighborhood_stability": 0.0,
            "max_drawdown": 1.0,
            "require_positive_net_expectancy": False,
        },
    )
    eng = HorizonResearchEngine(intraday_fixture, config=cfg)
    results = eng.sweep()
    r = results[0]
    assert r.trade_frequency["total_trades"] >= 1
    # long and/or short present for oscillating series
    assert r.trade_frequency["long_trades"] + r.trade_frequency["short_trades"] == r.trade_frequency[
        "total_trades"
    ]
    assert r.overtrading.get("note")


def test_cost_aware_comparison(intraday_fixture):
    low = HorizonResearchConfig(
        data_timeframes=["5m"],
        holding_bars=[5],
        instrument="INTRA",
        commission_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        robust_gates={
            "min_trades": 1,
            "min_oos_sharpe": -10,
            "min_oos_expectancy": -1e9,
            "min_neighborhood_stability": 0.0,
            "max_drawdown": 1.0,
            "require_positive_net_expectancy": False,
        },
    )
    high = HorizonResearchConfig(
        data_timeframes=["5m"],
        holding_bars=[5],
        instrument="INTRA",
        commission_bps=25.0,
        spread_bps=25.0,
        slippage_bps=25.0,
        robust_gates=dict(low.robust_gates),
    )
    r_low = HorizonResearchEngine(intraday_fixture, config=low).evaluate_spec(
        data_timeframe="5m", holding=5
    )
    r_high = HorizonResearchEngine(intraday_fixture, config=high).evaluate_spec(
        data_timeframe="5m", holding=5
    )
    assert r_high.costs["transaction_costs"] >= r_low.costs["transaction_costs"]
    assert r_high.metrics["net_sharpe"] <= r_low.metrics["net_sharpe"] + 1e-9


def test_walk_forward_oos_reported(daily_frame):
    cfg = HorizonResearchConfig(
        data_timeframes=["1D"],
        holding_bars=[1, 5],
        instrument="DEMO",
        train_frac=0.5,
        validation_frac=0.2,
        robust_gates={
            "min_trades": 1,
            "min_oos_sharpe": -10,
            "min_oos_expectancy": -1e9,
            "min_neighborhood_stability": 0.0,
            "max_drawdown": 1.0,
            "require_positive_net_expectancy": False,
        },
    )
    eng = HorizonResearchEngine(daily_frame, config=cfg)
    eng.sweep()
    for r in eng.results:
        if r.status == HorizonStatus.UNAVAILABLE:
            continue
        assert r.oos.get("evaluated") is True
        assert "train" in r.oos and "validation" in r.oos


def test_long_short_strategy_registry_and_transitions():
    StrategyRegistry.register(LongShortMomentumStrategy, overwrite=True)
    strat = StrategyRegistry.create("long_short_momentum", lookback=1, holding_bars=2)
    assert isinstance(strat, LongShortMomentumStrategy)

    class Ctx:
        universe = ["INTRA"]
        latest_prices: dict = {}
        strategy_state: dict = {}

    ctx = Ctx()
    strat.initialize(ctx)
    sides = []
    for i, px in enumerate([100, 101, 102, 99, 98, 100, 103, 101]):
        ctx.latest_prices = {"INTRA": float(px)}

        class Ev:
            payload = {"bars": {"INTRA": {"close": px}}}

        out = strat.on_features(Ev(), ctx)
        assert out is not None
        sides.append(out["side"])
        assert "target_weights" in out
    # multiple non-flat opportunities possible
    assert any(s in {"LONG", "SHORT"} for s in sides)
    end = strat.on_end(ctx)
    assert end["transitions"] is not None


def test_nifty_daily_horizon_if_present():
    path = Path("/home/ashish/qtb/data/nifty50/nifty50_daily.parquet")
    if not path.exists():
        pytest.skip("NIFTY daily parquet not present")
    df = pd.read_parquet(path)
    # normalize columns if needed
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for need in ("timestamp", "open", "high", "low", "close", "volume"):
        if need not in df.columns and need in cols:
            rename[cols[need]] = need
    if rename:
        df = df.rename(columns=rename)
    if "instrument" not in df.columns:
        df["instrument"] = "NIFTY50"
    if "timestamp" not in df.columns and "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cfg = HorizonResearchConfig(
        data_timeframes=["1m", "5m", "1h", "1D"],
        holding_bars=[1, 5, 20],
        instrument=str(df["instrument"].iloc[0]),
        train_end="2020-12-31",
        validation_end="2021-12-31",
        robust_gates={
            "min_trades": 3,
            "min_oos_sharpe": -10,
            "min_oos_expectancy": -1e9,
            "min_neighborhood_stability": 0.0,
            "max_drawdown": 1.0,
            "require_positive_net_expectancy": False,
        },
    )
    eng = HorizonResearchEngine(df, config=cfg)
    eng.sweep()
    report = eng.report()
    assert report["dataset"]["native_frequency"] in {"1D", "1d"} or parse_ok(report)
    statuses = {r["status"] for r in report["matrix"]}
    assert "UNAVAILABLE" in statuses
    assert any(row["data_timeframe"] == "1D" for row in report["matrix"])


def parse_ok(report: dict) -> bool:
    return "1D" in str(report["dataset"].get("native_frequency"))
