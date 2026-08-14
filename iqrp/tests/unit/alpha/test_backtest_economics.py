"""Signal/portfolio backtest, walk_forward, purged, embargo, nested; TC/turnover/capacity."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.alpha.backtesting.embargo import apply_embargo, embargo_splits
from iqrp.app.alpha.backtesting.nested_cv import nested_cv_splits
from iqrp.app.alpha.backtesting.portfolio_backtest import portfolio_backtest
from iqrp.app.alpha.backtesting.purged_cv import purge_train_indices, purged_kfold_splits
from iqrp.app.alpha.backtesting.signal_backtest import signal_backtest, signal_to_weights
from iqrp.app.alpha.backtesting.walk_forward import walk_forward_backtest, walk_forward_splits
from iqrp.app.alpha.economics.capacity import capacity_decay, estimate_capacity
from iqrp.app.alpha.economics.market_impact import market_impact_bps, market_impact_cost
from iqrp.app.alpha.economics.scalability import scalability_curve, scalability_report
from iqrp.app.alpha.economics.slippage import slippage_bps, slippage_cost
from iqrp.app.alpha.economics.transaction_costs import estimate_transaction_cost
from iqrp.app.alpha.economics.turnover import (
    annualized_turnover,
    average_turnover,
    turnover_series,
)


def test_signal_backtest_modes_and_costs(signal: np.ndarray, returns: np.ndarray) -> None:
    for mode in ("long_short", "long_only", "sign"):
        w = signal_to_weights(signal, mode=mode)
        assert w.shape == signal.shape

    gross = signal_backtest(signal, returns, cost_bps=0.0, returns_are_forward=False)
    net = signal_backtest(signal, returns, cost_bps=15.0, returns_are_forward=False)
    assert "gross_sharpe" in gross and "net_sharpe" in net
    # Transaction costs accounted: net mean <= gross mean under positive costs
    assert net["net_mean"] <= net["gross_mean"] + 1e-12
    assert net["total_cost"] >= 0.0
    # Costs reduce net vs zero-cost net
    assert net["net_mean"] <= gross["net_mean"] + 1e-9

    fwd_bt = signal_backtest(signal, returns, cost_bps=5.0, returns_are_forward=True)
    assert "look_ahead_guard" in fwd_bt or "n" in fwd_bt


def test_portfolio_backtest(rng: np.random.Generator) -> None:
    t, n = 120, 4
    rets = rng.normal(0, 0.01, size=(t, n))
    w = np.ones((t, n)) / n
    # Induce turnover
    w[60:, 0] = 0.5
    w[60:, 1:] = 0.5 / (n - 1)
    bt0 = portfolio_backtest(w, rets, cost_bps=0.0, returns_are_forward=True)
    bt_c = portfolio_backtest(w, rets, cost_bps=10.0, returns_are_forward=True)
    assert bt_c["net_mean"] <= bt_c["gross_mean"] + 1e-12
    assert bt_c["net_mean"] <= bt0["net_mean"] + 1e-9

    # 1D weights path
    bt1 = portfolio_backtest(np.ones(t) / 1.0, rets[:, 0], cost_bps=2.0)
    assert "n_assets" in bt1 or "net_sharpe" in bt1


def test_walk_forward(signal: np.ndarray, returns: np.ndarray) -> None:
    splits = list(
        walk_forward_splits(len(returns), train_size=100, test_size=30, gap=5, expanding=False)
    )
    assert len(splits) >= 1
    exp = list(
        walk_forward_splits(len(returns), train_size=80, test_size=25, gap=2, expanding=True)
    )
    assert len(exp) >= 1

    wf = walk_forward_backtest(signal, returns, train_size=100, test_size=30, gap=5, cost_bps=5.0)
    assert "n_folds" in wf
    assert wf["n_folds"] >= 1
    assert "folds" in wf


def test_purged_embargo_nested() -> None:
    n = 200
    folds = list(purged_kfold_splits(n, n_splits=4, purge=5))
    assert len(folds) == 4
    for tr, te in folds:
        # purge: train and test should not overlap
        assert len(set(tr) & set(te)) == 0

    purged = purge_train_indices(np.arange(100), np.arange(40, 60), purge=5, n=100)
    assert 45 not in set(purged)
    assert 30 in set(purged) or 70 in set(purged)

    train = np.arange(0, 80)
    test = np.arange(80, 100)
    emb = apply_embargo(train, test, embargo=5, purge=5)
    assert emb is not None

    esp = list(embargo_splits(n, n_splits=3, embargo=5, purge=5))
    assert len(esp) >= 1

    nested = list(nested_cv_splits(n, n_outer=3, n_inner=2, purge=3, embargo=3))
    assert len(nested) >= 1
    assert "outer_train" in nested[0]
    assert "inner_folds" in nested[0]
    assert nested[0].get("look_ahead_guard") is True or "purge" in nested[0]


def test_turnover_and_transaction_costs(rng: np.random.Generator) -> None:
    w = np.zeros((50, 3))
    w[:, 0] = 0.5
    w[:, 1] = 0.5
    w[25:, 0] = 0.2
    w[25:, 2] = 0.3
    w[25:, 1] = 0.5
    series = turnover_series(w, half=True)
    assert series.size == w.shape[0] - 1 or series.size == w.shape[0]
    avg = average_turnover(w)
    assert avg >= 0
    ann = annualized_turnover(w, periods_per_year=252)
    assert ann >= 0

    tc = estimate_transaction_cost(w[24], w[25], capital=1e6, cost_bps=5.0, prefer_portfolio=True)
    assert "total" in tc and tc["total"] >= 0
    tc_local = estimate_transaction_cost(
        w[24], w[25], capital=1e6, cost_bps=5.0, prefer_portfolio=False
    )
    assert tc_local["total"] >= 0


def test_slippage_impact_capacity_scalability() -> None:
    sb = slippage_bps(0.05, base_bps=1.0, participation_coeff=10.0, vol=0.02)
    assert sb >= 0
    sc = slippage_cost(1e5, participation=0.05)
    assert "total" in sc

    mb = market_impact_bps(0.05, impact_coeff=0.1, vol=0.02)
    assert float(np.asarray(mb).reshape(-1)[0]) >= 0
    mc = market_impact_cost(1e5, 0.05, impact_coeff=0.1, vol=0.02)
    assert "total" in mc

    cap = estimate_capacity(turnover=0.15, adv=5e7, max_participation=0.1)
    assert cap["max_capital"] > 0
    decay = capacity_decay(np.array([1e6, 5e7, 1e8]), max_capital=cap["max_capital"])
    assert np.all(decay <= 1.0 + 1e-9)

    capitals = np.array([1e5, 1e6, 1e7, 5e7])
    curve = scalability_curve(capitals=capitals, turnover=0.2, adv=5e7, gross_sharpe=1.5)
    assert "capitals" in curve or "net_sharpe" in curve
    report = scalability_report(turnover=0.2, adv=5e7, gross_sharpe=1.5, n_points=10)
    assert "capacity" in report or "max_viable_capital" in report


def test_empty_weights_turnover() -> None:
    w = np.ones((1, 2)) * 0.5
    assert np.isfinite(average_turnover(w))
    w2 = np.ones((5, 2)) * 0.5
    assert average_turnover(w2) >= 0.0
