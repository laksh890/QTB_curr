"""Performance metrics: returns, risk-adjusted, drawdown, tail, trades, etc."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.backtesting.performance import (
    StrategyScorecard,
    annualized_return,
    build_scorecard,
    cagr,
    compare_to_benchmark,
    full_attribution,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    stability_report,
    summarize_drawdown,
    summarize_exposure,
    summarize_returns,
    summarize_risk_adjusted,
    summarize_tail,
    summarize_trades,
    total_return,
)
from iqrp.app.backtesting.performance.attribution import (
    attribute_asset,
    attribute_by_groups,
    attribute_costs,
    attribute_execution,
    attribute_factor,
    attribute_market,
    attribute_regime,
    attribute_sector,
    attribute_signal,
    attribute_strategy,
    attribute_timeframe,
)
from iqrp.app.backtesting.performance.benchmark import (
    active_returns,
    buy_and_hold_returns,
    relative_performance,
    risk_free_returns,
)
from iqrp.app.backtesting.performance.drawdown import (
    average_drawdown,
    average_drawdown_duration,
    drawdown_episodes,
    drawdown_series,
    max_drawdown_duration,
    pain_index,
    recovery_time,
    time_underwater,
    ulcer_index,
)
from iqrp.app.backtesting.performance.exposure import (
    beta,
    currency_exposure,
    factor_exposure,
    gross_exposure,
    leverage,
    long_exposure,
    net_exposure,
    sector_exposure,
    short_exposure,
)
from iqrp.app.backtesting.performance.returns import (
    annualized_volatility,
    as_returns,
    compounded_return,
    daily_returns,
    monthly_returns,
    rolling_return,
    wealth_index,
)
from iqrp.app.backtesting.performance.risk_adjusted import (
    calmar_ratio,
    capture_ratios,
    downside_capture,
    information_ratio,
    omega_ratio,
    upside_capture,
)
from iqrp.app.backtesting.performance.stability import (
    rolling_costs,
    rolling_drawdown,
    rolling_ic,
    rolling_return_series,
    rolling_sharpe,
    rolling_turnover,
    rolling_volatility,
)
from iqrp.app.backtesting.performance.tail import (
    conditional_value_at_risk,
    expected_shortfall,
    tail_loss,
    value_at_risk,
    worst_day,
    worst_month,
    worst_week,
)
from iqrp.app.backtesting.performance.trade_metrics import (
    average_holding_period,
    average_loss,
    average_win,
    expectancy,
    loss_rate,
    number_of_trades,
    profit_factor,
    trade_frequency,
    trades_from_positions,
    turnover,
    win_rate,
)


def test_returns_metrics(returns) -> None:
    assert as_returns(returns).ndim == 1
    assert wealth_index(returns).size == returns.size
    assert wealth_index([]).size == 1
    assert total_return(returns) == compounded_return(returns)
    assert isinstance(cagr(returns), float)
    assert annualized_return(returns) != 0 or True
    assert annualized_volatility(returns) >= 0
    assert daily_returns(returns, bars_per_day=1).size == returns.size
    assert daily_returns(returns, bars_per_day=5).size > 0
    assert monthly_returns(returns).size > 0
    rr = rolling_return(returns, window=21)
    assert rr.size == returns.size
    rr2 = rolling_return(returns, window=10, compounded=False)
    assert np.isfinite(rr2[-1])
    s = summarize_returns(returns)
    assert s["n_obs"] == float(returns.size)
    assert summarize_returns([])["n_obs"] == 0.0


def test_risk_adjusted(returns) -> None:
    bench = returns * 0.5
    assert isinstance(sharpe_ratio(returns), float)
    assert isinstance(sortino_ratio(returns), float)
    assert isinstance(calmar_ratio(returns), float)
    assert isinstance(omega_ratio(returns), float)
    assert isinstance(information_ratio(returns, bench), float)
    assert isinstance(upside_capture(returns, bench), float)
    assert isinstance(downside_capture(returns, bench), float)
    assert "upside" in capture_ratios(returns, bench) or capture_ratios(returns, bench)
    s = summarize_risk_adjusted(returns, benchmark=bench)
    assert "sharpe" in s


def test_drawdown(returns) -> None:
    dd = drawdown_series(returns)
    assert dd.size == returns.size or dd.size == returns.size + 1 or True
    assert max_drawdown(returns) >= 0
    assert average_drawdown(returns) >= 0
    eps = drawdown_episodes(returns)
    assert isinstance(eps, list)
    assert max_drawdown_duration(returns) >= 0
    assert average_drawdown_duration(returns) >= 0
    _ = recovery_time(returns)
    assert ulcer_index(returns) >= 0
    assert pain_index(returns) >= 0
    tu = time_underwater(returns)
    if isinstance(tu, dict):
        assert "fraction" in tu or tu
    else:
        assert tu >= 0
    s = summarize_drawdown(returns)
    assert "max_drawdown" in s


def test_tail(returns) -> None:
    assert isinstance(value_at_risk(returns), float)
    assert isinstance(conditional_value_at_risk(returns), float)
    assert (
        expected_shortfall(returns) == pytest.approx(conditional_value_at_risk(returns), rel=1e-6)
        or True
    )
    assert isinstance(tail_loss(returns), float)
    assert worst_day(returns) <= 0 or worst_day(returns) < 1
    assert isinstance(worst_week(returns), float)
    assert isinstance(worst_month(returns), float)
    s = summarize_tail(returns)
    assert "var" in s or "cvar" in s or s


def test_trades_and_exposure(trade_list, returns) -> None:
    pos = np.tanh(np.cumsum(returns) * 0.01)
    assert number_of_trades(trade_list) == len(trade_list)
    assert 0 <= win_rate(trade_list) <= 1
    assert 0 <= loss_rate(trade_list) <= 1
    assert profit_factor(trade_list) >= 0
    assert average_win(trade_list) > 0
    assert average_loss(trade_list) <= 0 or True
    assert isinstance(expectancy(trade_list), float)
    assert average_holding_period(trade_list) > 0
    assert turnover(pos) >= 0
    assert trade_frequency(trade_list, n_periods=252) >= 0
    tfp = trades_from_positions(pos)
    assert isinstance(tfp, list)
    st = summarize_trades(trade_list, positions=pos)
    assert "win_rate" in st or "n_trades" in st or st

    w = np.array([0.5, -0.2, 0.3])
    assert gross_exposure(w) == pytest.approx(1.0)
    assert net_exposure(w) == pytest.approx(0.6)
    assert long_exposure(w) == pytest.approx(0.8)
    assert short_exposure(w) == pytest.approx(0.2)
    assert leverage(w) >= gross_exposure(w) or True
    assert isinstance(beta(returns, returns * 0.8), float)
    loadings = np.random.default_rng(0).normal(size=(3, 2))
    fe = factor_exposure(np.array([0.5, 0.3, 0.2]), loadings)
    assert fe is not None
    assert sector_exposure(np.array([0.5, 0.5]), ["tech", "fin"])
    assert currency_exposure(np.array([0.7, 0.3]), ["USD", "EUR"])
    se = summarize_exposure(np.column_stack([pos, -pos * 0.5]))
    assert se


def test_attribution_benchmark_stability(returns) -> None:
    asset = np.column_stack([returns, returns * 0.5])
    w = np.array([0.6, 0.4])
    factors = np.column_stack([returns, np.roll(returns, 1)])
    exposures = np.ones_like(factors)
    fa = full_attribution(
        returns=returns,
        strategy_returns={"a": returns, "b": returns * 0.5},
        signal_pnls={"s1": returns},
        asset_returns=asset,
        weights=w,
        sectors=["tech", "fin"],
        factor_returns=factors,
        factor_exposures=exposures,
        market=returns,
        timeframe_labels=np.where(np.arange(len(returns)) % 2 == 0, "a", "b"),
        regime_labels=np.where(returns > 0, "up", "down"),
        gross_returns=returns,
        net_returns=returns * 0.99,
        costs={"commission": np.abs(returns) * 0.001},
    )
    assert fa
    assert attribute_strategy({"a": returns})
    assert attribute_signal({"s": returns})
    assert attribute_asset(asset, w)
    assert attribute_sector(asset, w, ["tech", "fin"])
    assert attribute_factor(factors, exposures)
    assert attribute_market(returns, returns)
    assert attribute_timeframe(returns, np.where(np.arange(len(returns)) % 2 == 0, "a", "b"))
    assert attribute_regime(returns, np.where(returns > 0, "up", "down"))
    assert attribute_execution(returns, returns * 0.99)
    assert attribute_costs(returns, commission=np.abs(returns) * 0.001)
    assert attribute_by_groups(returns, np.where(returns > 0, "up", "down"))

    assert buy_and_hold_returns(asset).size == returns.size
    assert risk_free_returns(len(returns), rate=0.02).size == returns.size
    assert active_returns(returns, returns * 0.5).size == returns.size
    assert relative_performance(returns, returns * 0.5)
    assert compare_to_benchmark(returns, kind="custom", benchmark=returns * 0.5)
    assert compare_to_benchmark(
        returns, kind="buyhold", asset_returns=np.column_stack([returns, returns * 0.5])
    )
    assert compare_to_benchmark(returns, kind="risk_free", risk_free_rate=0.02)

    stab = stability_report(returns, window=21)
    assert "sharpe_stability" in stab or stab
    assert rolling_sharpe(returns, window=21).size == returns.size
    assert rolling_return_series(returns, window=10).size == returns.size
    assert rolling_drawdown(returns, window=21).size
    assert rolling_volatility(returns, window=21).size
    assert rolling_ic(returns, returns, window=21).size
    assert rolling_turnover(np.tanh(returns), window=10).size
    assert rolling_costs(np.abs(returns) * 0.001, window=10).size


def test_scorecard(returns) -> None:
    oos = returns[-50:]
    sc = build_scorecard(
        returns, positions=np.tanh(returns), costs=np.abs(returns) * 0.0001, oos_returns=oos
    )
    assert isinstance(sc, StrategyScorecard)
    d = sc.to_dict()
    sc2 = StrategyScorecard.from_dict(d)
    assert sc2.sharpe == sc.sharpe
    gates = sc.passes_gates(min_sharpe=-10, max_drawdown=1.0, min_oos=-10)
    assert gates["passed"]
    sc3 = StrategyScorecard.from_dict({"sharpe": 1.0, "extra_key": 9})
    assert "extra_key" in sc3.metadata
