"""Unit tests for horizon research primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iqrp.app.backtesting.data.synthetic import generate_synthetic_ohlcv
from iqrp.app.backtesting.horizon.availability import (
    check_horizon_availability,
    detect_native_frequency,
    filter_available_timeframes,
)
from iqrp.app.backtesting.horizon.costs import apply_cost_drag, gross_vs_net_sharpe
from iqrp.app.backtesting.horizon.neighborhood import neighborhood_robustness
from iqrp.app.backtesting.horizon.parse import can_derive, parse_holding, parse_timeframe
from iqrp.app.backtesting.horizon.ranking import (
    classify_horizon,
    compute_horizon_research_score,
)
from iqrp.app.backtesting.horizon.resampling import UnavailableFrequencyError, resample_ohlcv
from iqrp.app.backtesting.horizon.trade_analytics import (
    classify_side,
    enrich_trades_with_holding,
    trade_frequency_report,
)
from iqrp.app.backtesting.horizon.turnover import turnover_report
from iqrp.app.backtesting.horizon.types import HorizonStatus


def test_parse_timeframe_common_tokens():
    assert parse_timeframe("1m").seconds == 60
    assert parse_timeframe("5m").label == "5m"
    assert parse_timeframe("15m").seconds == 900
    assert parse_timeframe("30m").seconds == 1800
    assert parse_timeframe("1h").seconds == 3600
    assert parse_timeframe("4h").seconds == 14400
    assert parse_timeframe("1D").label == "1D"
    assert parse_timeframe("daily").label == "1D"


def test_parse_holding_bars_and_time():
    h = parse_holding("5bar")
    assert h.bars == 5
    assert str(h) == "5bar"
    h2 = parse_holding(10)
    assert h2.bars == 10
    h3 = parse_holding("30m", bar_seconds=300.0)
    assert h3.seconds == 1800
    assert h3.bars == 6


def test_can_derive_and_unavailable():
    assert can_derive("1D", "1D")
    assert can_derive("5m", "1h")
    assert not can_derive("1D", "5m")
    gate = check_horizon_availability("1D", "1m")
    assert gate["available"] is False
    assert gate["status"] == HorizonStatus.UNAVAILABLE.value


def test_resample_refuses_finer_than_native():
    frame = generate_synthetic_ohlcv(n_days=40, freq="1d", seed=3, instruments=["X"])
    native = detect_native_frequency(frame)
    assert native.seconds >= 86400 - 1
    with pytest.raises(UnavailableFrequencyError):
        resample_ohlcv(frame, "5m", native=native)
    daily = resample_ohlcv(frame, "1D", native=native)
    assert len(daily) == len(frame.loc[frame["instrument"] == "X"])


def test_filter_available_timeframes_daily_native():
    out = filter_available_timeframes("1D", ["1m", "5m", "1h", "1D"])
    assert out["available"] == ["1D"]
    assert len(out["unavailable"]) == 3


def test_trade_frequency_and_holding():
    trades = [
        {
            "side": "buy",
            "pnl": 1.0,
            "entry_time": "2020-01-02T10:00:00Z",
            "exit_time": "2020-01-02T11:00:00Z",
        },
        {
            "side": "sell",
            "pnl": -0.5,
            "entry_time": "2020-01-03T10:00:00Z",
            "exit_time": "2020-01-03T12:00:00Z",
        },
        {
            "side": "long",
            "pnl": 0.2,
            "entry_time": "2020-01-03T14:00:00Z",
            "exit_time": "2020-01-03T15:00:00Z",
        },
    ]
    enriched = enrich_trades_with_holding(trades)
    assert enriched[0]["holding_seconds"] == 3600
    assert classify_side("short").value == "SHORT"
    rep = trade_frequency_report(trades)
    assert rep["total_trades"] == 3
    assert rep["long_trades"] == 2
    assert rep["short_trades"] == 1
    assert rep["maximum_trades_per_day"] >= 2
    assert rep["average_holding_period_seconds"] is not None


def test_cost_attribution_erodes_edge():
    rng = np.random.default_rng(0)
    gross = rng.normal(0.001, 0.01, size=200)
    # extreme turnover → costs kill edge
    to = np.full(200, 2.0)
    cost = apply_cost_drag(
        gross,
        commission_bps=10,
        spread_bps=10,
        slippage_bps=10,
        turnover_per_period=to,
    )
    gn = gross_vs_net_sharpe(cost["gross_returns"], cost["net_returns"])
    assert "gross_sharpe" in gn and "net_sharpe" in gn
    assert cost["transaction_costs"] > 0


def test_turnover_report():
    pos = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    rep = turnover_report(pos, periods_per_day=1.0, net_pnl=1.0, net_alpha=0.01)
    assert rep["annualized_turnover"] > 0
    assert rep["pnl_per_unit_turnover"] is not None


def test_horizon_ranking_not_max_return():
    metrics = {
        "net_sharpe": 1.5,
        "gross_sharpe": 1.6,
        "expectancy_per_trade": 0.01,
        "maximum_drawdown": 0.1,
        "trade_count": 40,
        "total_return_gross": 0.2,
        "stability": 0.8,
    }
    scored = compute_horizon_research_score(
        metrics,
        oos={"net_sharpe": 1.2, "evaluated": True},
        costs={"transaction_costs": 0.01},
        turnover={"annualized_turnover": 5.0},
        neighborhood={"stability_score": 0.9, "fragile": False},
        multiple_testing={"n_configurations_tested": 10, "confidence": 0.7},
    )
    assert 0.0 <= scored["score"] <= 1.0
    assert "weights" in scored


def test_classify_cost_inefficient_and_fragile():
    st, _ = classify_horizon(
        {"net_sharpe": 0.1, "gross_sharpe": 2.0, "trade_count": 20, "expectancy_per_trade": 0.01, "maximum_drawdown": 0.05},
        costs={"cost_eroded_edge": True},
    )
    assert st == HorizonStatus.COST_INEFFICIENT
    st2, _ = classify_horizon(
        {"net_sharpe": 1.0, "gross_sharpe": 1.1, "trade_count": 20, "expectancy_per_trade": 0.01, "maximum_drawdown": 0.05},
        oos={"evaluated": True, "net_sharpe": 1.0, "expectancy_per_trade": 0.01},
        neighborhood={"fragile": True, "stability_score": 0.1},
    )
    assert st2 == HorizonStatus.FRAGILE


def test_neighborhood_robustness_flags_spike():
    rows = [
        {"spec": {"data_timeframe": "5m"}, "metrics": {"net_sharpe": 0.1}},
        {"spec": {"data_timeframe": "15m"}, "metrics": {"net_sharpe": 3.0}},
        {"spec": {"data_timeframe": "30m"}, "metrics": {"net_sharpe": 0.05}},
    ]
    out = neighborhood_robustness(rows, max_ratio=4.0)
    assert out["15m"]["fragile"] is True


def test_intraday_fixture_downsample():
    # Deterministic software fixture only — not a profitability claim
    start = pd.Timestamp("2020-01-02", tz="UTC")
    ts = pd.date_range(start, periods=120, freq="5min", tz="UTC")
    close = 100 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.001, size=len(ts)))
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "instrument": "FX",
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1000.0,
        }
    )
    native = detect_native_frequency(frame)
    assert native.seconds == 300
    h1 = resample_ohlcv(frame, "15m", native=native)
    assert len(h1) < len(frame)
    with pytest.raises(UnavailableFrequencyError):
        resample_ohlcv(frame, "1m", native=native)
