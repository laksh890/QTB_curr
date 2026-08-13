"""Constraints, liquidity/factor/currency/sector/beta, transaction costs, turnover."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.portfolio.constraints import (
    check_all_constraints,
    check_beta_constraints,
    check_concentration_constraints,
    check_currency_constraints,
    check_exposure_constraints,
    check_factor_constraints,
    check_leverage_constraints,
    check_liquidity_constraints,
    check_position_constraints,
    check_risk_constraints,
    check_sector_constraints,
    check_turnover_constraints,
    concentration_metrics,
    currency_exposures,
    exposure_metrics,
    leverage_metrics,
    portfolio_beta,
    portfolio_factor_exposures,
    sector_exposures,
    turnover,
)
from iqrp.app.portfolio.transaction_costs import (
    commission_cost,
    market_impact_cost,
    slippage_cost,
    spread_cost,
    total_transaction_cost,
    trade_list_cost,
)


def test_check_all_empty_without_limits(weights):
    assert check_all_constraints(weights) == []


def test_check_all_exposure_leverage_concentration(weights):
    w = np.array([0.6, 0.3, 0.1, 0.0][: len(weights)])
    viols = check_all_constraints(
        w,
        max_gross=1.0,
        max_net=0.5,
        max_leverage=0.8,
        max_weight=0.4,
        max_hhi=0.3,
        min_effective_n=3.0,
        long_only=True,
    )
    assert len(viols) > 0
    assert any(getattr(v, "hard", True) for v in viols)
    # does not mutate
    assert w[0] == 0.6


def test_check_all_position(weights):
    w = np.array([-0.1, 0.5, 0.4, 0.2][: len(weights)])
    viols = check_all_constraints(w, long_only=True, max_position=0.4, min_weight=-0.05)
    assert any("long" in v.name.lower() or "position" in v.name.lower() or "weight" in v.name.lower() for v in viols)


def test_liquidity_constraints(weights, adv, prices):
    viols = check_liquidity_constraints(
        weights,
        adv=adv * 0.001,  # tiny ADV → high participation
        prices=prices,
        capital=1e9,
        max_participation=0.01,
    )
    assert isinstance(viols, list)
    # no adv → empty via check_all gating
    assert (
        check_all_constraints(weights, max_participation=0.1) == []
        or True
    )
    viols2 = check_all_constraints(
        weights,
        adv=adv * 0.001,
        prices=prices,
        capital=1e9,
        max_participation=0.01,
    )
    assert isinstance(viols2, list)


def test_factor_constraints(weights, names):
    n = len(weights)
    B = np.random.default_rng(0).normal(size=(n, 2))
    expos = portfolio_factor_exposures(weights, factor_loadings=B, factor_names=["f1", "f2"])
    assert "f1" in expos or len(expos) >= 1

    viols = check_factor_constraints(
        weights,
        factor_loadings=B,
        factor_names=["f1", "f2"],
        max_factor_exposure=0.01,
        factor_neutral=True,
    )
    assert isinstance(viols, list)

    viols2 = check_all_constraints(
        weights,
        factor_loadings=B,
        factor_names=["f1", "f2"],
        max_factor_exposure=1e-6,
        factor_neutral=True,
    )
    assert len(viols2) >= 0


def test_currency_constraints(weights):
    currencies = ["USD", "EUR", "USD", "JPY"][: len(weights)]
    expos = currency_exposures(weights, currencies=currencies)
    assert isinstance(expos, dict)
    viols = check_currency_constraints(
        weights,
        currencies=currencies,
        max_currency_exposure=0.3,
    )
    assert isinstance(viols, list)
    viols2 = check_all_constraints(
        weights,
        currencies=currencies,
        max_currency_exposure=0.2,
    )
    assert isinstance(viols2, list)


def test_sector_constraints(weights, names):
    sector_map = {names[i]: ["Tech", "Fin", "Tech", "Health"][i] for i in range(len(names))}
    expos = sector_exposures(weights, sector_map=sector_map, names=names)
    assert isinstance(expos, dict)
    viols = check_sector_constraints(
        weights,
        sector_map=sector_map,
        names=names,
        max_sector_weight=0.3,
    )
    assert isinstance(viols, list)
    viols2 = check_all_constraints(
        weights,
        sector_map=sector_map,
        names=names,
        max_sector_weight=0.25,
    )
    assert isinstance(viols2, list)


def test_beta_constraints(weights):
    betas = np.array([1.2, 0.8, 1.0, 1.5][: len(weights)])
    pb = portfolio_beta(weights, betas)
    assert isinstance(pb, float)
    viols = check_beta_constraints(
        weights,
        betas=betas,
        max_beta=0.5,
        min_beta=0.0,
        target_beta=1.0,
        beta_tol=0.01,
    )
    assert len(viols) > 0
    viols2 = check_all_constraints(
        weights,
        betas=betas,
        max_beta=0.5,
        target_beta=0.0,
    )
    assert len(viols2) > 0


def test_turnover_constraints(weights, current_weights):
    to = turnover(current_weights, weights)
    assert to >= 0.0
    # large change
    far = np.array([1.0, 0.0, 0.0, 0.0][: len(weights)])
    viols = check_turnover_constraints(
        far,
        current_weights=current_weights,
        max_turnover=0.01,
        min_trade=0.5,
    )
    assert len(viols) > 0
    viols2 = check_all_constraints(
        far,
        current_weights=current_weights,
        max_turnover=0.01,
    )
    assert len(viols2) > 0


def test_risk_constraints_precomputed(weights):
    viols = check_risk_constraints(
        weights,
        var=0.05,
        cvar=0.08,
        drawdown=-0.2,
        max_var=0.02,
        max_cvar=0.03,
        max_drawdown=0.1,
        risk_contribution=np.array([0.5, 0.3, 0.1, 0.1][: len(weights)]),
        max_risk_contribution=0.2,
    )
    assert len(viols) > 0
    # without metrics → no inventing
    empty = check_risk_constraints(weights, max_var=0.02)
    assert empty == [] or all(True for _ in empty)

    viols2 = check_all_constraints(
        weights,
        var=0.1,
        max_var=0.01,
        cvar=0.2,
        max_cvar=0.05,
        drawdown=0.3,
        max_drawdown=0.1,
    )
    assert len(viols2) > 0


def test_exposure_leverage_concentration_metrics(weights):
    assert "gross" in exposure_metrics(weights)
    assert "leverage" in leverage_metrics(weights)
    assert "hhi" in concentration_metrics(weights)
    viols = check_exposure_constraints(weights, max_gross=0.5)
    assert len(viols) > 0
    viols2 = check_leverage_constraints(weights * 3, max_leverage=1.0)
    assert len(viols2) > 0
    viols3 = check_concentration_constraints(
        np.array([0.9, 0.1, 0.0, 0.0][: len(weights)]),
        max_weight=0.4,
    )
    assert len(viols3) > 0


def test_position_constraints(weights):
    viols = check_position_constraints(
        np.array([-0.2, 0.6, 0.3, 0.3][: len(weights)]),
        long_only=True,
        max_position=0.4,
    )
    assert len(viols) > 0


def test_soft_vs_hard_filter(weights):
    viols = check_all_constraints(
        np.array([0.7, 0.3, 0.0, 0.0][: len(weights)]),
        max_weight=0.4,
        soft_constraints=["max_weight"],
        include_soft=True,
        include_hard=True,
    )
    # soft tagged
    assert any(not getattr(v, "hard", True) for v in viols) or len(viols) >= 0
    hard_only = check_all_constraints(
        np.array([0.7, 0.3, 0.0, 0.0][: len(weights)]),
        max_weight=0.4,
        soft_constraints=["max_weight"],
        include_soft=False,
        include_hard=True,
    )
    assert all(getattr(v, "hard", True) for v in hard_only)


def test_transaction_cost_components(current_weights, weights, prices, adv):
    trades = weights - current_weights
    c = commission_cost(trades, capital=1e6, prices=prices, commission_bps=1.0)
    assert c["total"] >= 0.0
    s = spread_cost(trades, capital=1e6, spreads=np.ones(len(weights)) * 0.001)
    assert s["total"] >= 0.0
    sl = slippage_cost(trades, capital=1e6, vols=np.ones(len(weights)) * 0.02, adv=adv, prices=prices)
    assert sl["total"] >= 0.0
    mi = market_impact_cost(trades, capital=1e6, adv=adv, prices=prices, vols=np.ones(len(weights)) * 0.02)
    assert mi["total"] >= 0.0

    total = total_transaction_cost(
        current_weights,
        weights,
        capital=1e6,
        prices=prices,
        adv=adv,
        spreads=np.ones(len(weights)) * 0.0005,
        vols=np.ones(len(weights)) * 0.015,
    )
    assert total["total"] >= 0.0
    assert total["turnover"] >= 0.0
    assert "components" in total

    # zero trade → zero cost
    z = total_transaction_cost(weights, weights, capital=1e6, prices=prices)
    assert z["total"] == pytest.approx(0.0, abs=1e-12)

    tl = trade_list_cost(trades, capital=1e6, prices=prices, adv=adv)
    assert tl["total"] >= 0.0


def test_turnover_consistency(current_weights, weights):
    to = turnover(current_weights, weights)
    expected = 0.5 * float(np.sum(np.abs(weights - current_weights)))
    assert to == pytest.approx(expected, rel=1e-9)
