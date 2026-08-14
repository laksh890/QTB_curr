"""Covariance, expected returns, BL, signal→weight, target positions, rebalance."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.portfolio.config import PortfolioSettings
from iqrp.app.portfolio.construction import (
    PortfolioConstructor,
    RebalanceBands,
    build_target_weights,
    plan_rebalance,
    signals_to_raw_weights,
)
from iqrp.app.portfolio.construction.rebalance import apply_rebalance_bands, evaluate_triggers
from iqrp.app.portfolio.construction.signal_to_weight import (
    rank_weights,
    softmax_weights,
    zscore_weights,
)
from iqrp.app.portfolio.construction.target_positions import weights_to_positions
from iqrp.app.portfolio.construction.target_weights import TargetWeights
from iqrp.app.portfolio.covariance import (
    ewma_covariance,
    factor_covariance,
    ledoit_wolf_covariance,
    robust_covariance,
    sample_covariance,
    shrinkage_covariance,
)
from iqrp.app.portfolio.expected_returns import (
    black_litterman_posterior,
    forecast_expected_returns,
    historical_expected_returns,
    shrinkage_expected_returns,
)
from iqrp.app.portfolio.expected_returns.black_litterman import equilibrium_returns
from iqrp.app.portfolio.expected_returns.shrinkage import james_stein_shrinkage


def test_sample_ewma_shrinkage_lw_robust(returns):
    for fn, kwargs in (
        (sample_covariance, {}),
        (ewma_covariance, {"lambda_": 0.94}),
        (shrinkage_covariance, {}),
        (ledoit_wolf_covariance, {}),
        (robust_covariance, {"seed": 42, "n_trials": 8}),
    ):
        out = fn(returns, **kwargs)
        mat = np.asarray(out["matrix"])
        assert mat.shape == (returns.shape[1], returns.shape[1])
        # PSD-ish
        eig = np.linalg.eigvalsh(0.5 * (mat + mat.T))
        assert eig.min() >= -1e-8


def test_factor_covariance(returns):
    n = returns.shape[1]
    B = np.eye(n)[:, :2]
    out = factor_covariance(
        factor_loadings=B,
        asset_returns=returns,
        factor_returns=returns[:, :2],
    )
    assert np.asarray(out["matrix"]).shape == (n, n)


def test_historical_and_shrinkage_expected_returns(returns, names):
    h = historical_expected_returns(returns, names=names)
    assert len(h["mu"]) == len(names)
    assert h.get("research_only") is True or "warning" in h

    s = shrinkage_expected_returns(returns=returns, names=names)
    assert len(s["mu"]) == len(names)

    js = james_stein_shrinkage(np.mean(returns, axis=0), names=names)
    assert "mu" in js


def test_forecast_expected_returns_confidence_shrink(forecasts, names):
    high = forecast_expected_returns(forecasts, confidence=np.ones(len(names)), names=names)
    low = forecast_expected_returns(
        forecasts,
        confidence=np.zeros(len(names)),
        prior=np.zeros(len(names)),
        names=names,
    )
    # low confidence → shrinks toward prior (zeros) — cannot invent certainty
    assert np.linalg.norm(low["mu"]) <= np.linalg.norm(high["mu"]) + 1e-12


def test_black_litterman_with_and_without_views(cov, names):
    n = len(names)
    mw = np.ones(n) / n
    eq = equilibrium_returns(cov, mw, risk_aversion=1.0)
    assert eq.shape == (n,)

    no_views = black_litterman_posterior(cov=cov, market_weights=mw, names=names)
    assert len(no_views["mu"]) == n

    P = np.zeros((1, n))
    P[0, 0] = 1.0
    Q = np.array([0.01])
    with_views = black_litterman_posterior(
        cov=cov, market_weights=mw, P=P, Q=Q, names=names, tau=0.05
    )
    assert with_views.get("n_views", 1) >= 1
    assert len(with_views["mu"]) == n


def test_signals_to_raw_weights_methods(signals, names):
    for method in ("zscore", "rank", "softmax", "proportional", "identity"):
        out = signals_to_raw_weights(signals, method=method, names=names, long_only=True)
        assert abs(sum(out["weights"]) - out["budget"]) < 1e-5 or abs(sum(out["weights"])) < 1e-8
        assert out["method"] == method

    with pytest.raises(ValueError):
        signals_to_raw_weights(signals, method="unknown_xyz")

    # wrappers
    assert len(rank_weights(signals)["weights"]) == len(signals)
    assert len(zscore_weights(signals)["weights"]) == len(signals)
    assert len(softmax_weights(signals)["weights"]) == len(signals)


def test_signals_all_negative_long_only():
    out = signals_to_raw_weights(
        np.array([-1.0, -2.0, -0.5]), method="proportional", long_only=True
    )
    assert abs(sum(out["weights"]) - 1.0) < 1e-5


def test_build_target_weights_and_cash(names, weights):
    tw = build_target_weights(weights, names=names, method="test")
    assert isinstance(tw, TargetWeights)
    assert tw.to_dict()["names"] == list(names)
    cash = TargetWeights.cash()
    assert len(cash.weights) == 0 or all(abs(w) < 1e-12 for w in cash.weights)
    eq = TargetWeights.equal_weight(names)
    assert abs(sum(eq.weights) - 1.0) < 1e-9


def test_weights_to_positions(names, weights, prices):
    tp = weights_to_positions(
        weights,
        capital=1_000_000.0,
        prices=prices,
        names=names,
        lot_sizes=np.ones(len(names)),
        round_lots=True,
    )
    assert tp.capital == 1_000_000.0
    assert len(tp.positions) == len(names)
    d = tp.to_dict()
    assert "positions" in d


def test_constructor(signals, names, prices, portfolio_settings):
    ctor = PortfolioConstructor(portfolio_settings)
    tw = ctor.signals_to_weights(signals, method="zscore", names=names)
    assert len(tw.weights) == len(names)
    tw2 = ctor.build_target_weights(tw.weights, names=names)
    pos = ctor.build_positions(tw2, capital=1e6, prices=prices)
    assert pos is not None


def test_rebalance_bands_and_triggers(current_weights, weights, names):
    trades = apply_rebalance_bands(
        current_weights,
        weights,
        absolute=0.5,  # large band → no trades
        relative=0.0,
        min_trade=0.0,
    )
    assert np.allclose(trades, 0.0)

    trades2 = apply_rebalance_bands(
        current_weights,
        np.array([1.0, 0.0, 0.0, 0.0][: len(weights)]),
        absolute=0.0,
        relative=0.0,
        min_trade=0.0,
    )
    assert np.sum(np.abs(trades2)) > 0

    bands = RebalanceBands(absolute=0.01, relative=0.05, min_trade=0.001)
    plan = plan_rebalance(
        current_weights,
        weights,
        bands=bands,
        names=names,
        force=True,
    )
    assert plan.should_rebalance is True
    assert plan.to_dict()["should_rebalance"] is True

    triggers = evaluate_triggers(
        current_weights=current_weights,
        target_weights=weights,
        turnover_threshold=0.0,
        force=True,
    )
    assert any(t.fired for t in triggers)


def test_rebalance_schedule_like_no_force(current_weights, names):
    plan = plan_rebalance(
        current_weights,
        current_weights,  # no drift
        bands=RebalanceBands(absolute=0.05, relative=0.1, min_trade=0.01),
        names=names,
        force=False,
        turnover_threshold=0.5,
    )
    # identical → typically no rebalance unless trigger
    assert plan.turnover == pytest.approx(0.0, abs=1e-12)
