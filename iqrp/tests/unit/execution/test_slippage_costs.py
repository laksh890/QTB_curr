"""Slippage estimation, realized slippage, pre/post trade costs, IS attribution."""

from __future__ import annotations

from iqrp.app.execution.slippage import (
    ExecutionSlippageModel,
    HistoricalSlippageModel,
    HistoricalSlippageRecord,
    compare_expected_realized,
    effective_spread_bps,
    estimate_slippage,
    historical_slippage_bps,
    impact_curve,
    liquidity_slippage,
    market_impact,
    nonlinear_impact,
    path_impact,
    realized_slippage,
    spread_slippage,
    volatility_slippage,
)
from iqrp.app.execution.transaction_costs import (
    borrow_cost,
    commission_cost,
    exchange_fees,
    financing_cost,
    market_impact_cost,
    post_trade_cost_analysis,
    pre_trade_cost_estimate,
    slippage_cost,
    spread_cost,
)


def test_estimate_slippage_components():
    est = estimate_slippage(
        side="buy",
        quantity=1_000.0,
        mid=100.0,
        spread=0.02,
        adv=1e6,
        volatility=0.02,
    )
    assert est["expected_slippage_bps"] >= 0
    assert "components" in est
    assert est["quantity"] == 1000.0

    nl = estimate_slippage(
        side="sell",
        quantity=500.0,
        mid=100.0,
        spread=0.04,
        adv=5e5,
        use_nonlinear=True,
        nonlinear_exponent=0.6,
    )
    assert nl["expected_slippage"] >= 0


def test_realized_slippage_and_compare():
    fills = [
        {"quantity": 50.0, "price": 100.05},
        {"quantity": 50.0, "price": 100.10},
    ]
    real = realized_slippage(fills, side="buy", arrival_price=100.0)
    assert real["realized_slippage"] > 0 or "realized_slippage_bps" in real
    cmp_ = compare_expected_realized(
        fills,
        side="buy",
        quantity=100.0,
        mid=100.0,
        arrival_price=100.0,
        spread=0.02,
        adv=1e6,
    )
    assert "expected" in cmp_ and "realized" in cmp_


def test_market_impact_path_and_nonlinear():
    mi = market_impact(side="buy", quantity=1000, mid=100.0, adv=1e6, volatility=0.02, spread=0.02)
    assert mi["temporary_impact"] >= 0
    assert mi["permanent_impact"] >= 0
    curve = impact_curve([0.01, 0.05, 0.1], mid=100.0, volatility=0.02)
    assert len(curve) >= 1
    nl = nonlinear_impact(quantity=1000, mid=100.0, adv=1e6, volatility=0.02)
    assert nl["temporary_impact"] >= 0
    path = path_impact(
        [100.0, 100.1, 100.2],
        [1e6, 1e6, 1e6],
        [100.0, 100.0, 100.0],
        [0.02, 0.02, 0.02],
    )
    assert len(list(path)) == 3


def test_spread_vol_liquidity_helpers():
    assert spread_slippage(mid=100.0, spread=0.02, side="buy")["slippage"] >= 0
    assert volatility_slippage(mid=100.0, volatility=0.02, horizon_seconds=60)["slippage"] >= 0
    assert liquidity_slippage(mid=100.0, quantity=1000, adv=1e6)["slippage"] >= 0
    assert effective_spread_bps(side="buy", fill_price=100.05, mid=100.0) >= 0


def test_historical_slippage_model():
    recs = [
        HistoricalSlippageRecord(participation=0.01, slippage_bps=5.0, side="buy"),
        HistoricalSlippageRecord(participation=0.02, slippage_bps=8.0, side="buy"),
    ]
    model = HistoricalSlippageModel(records=recs)
    pred = model.estimate_bps(quantity=150, adv=1e6)
    assert pred["expected_slippage_bps"] >= 0
    assert model.calibrate_linear()["n_obs"] == 2.0
    bps = historical_slippage_bps(150.0, 1e6, records=recs)
    assert bps >= 0


def test_execution_slippage_model():
    model = ExecutionSlippageModel(impact_coeff=0.1)
    br = model.estimate(
        side="buy", quantity=100, mid=100.0, spread=0.02, adv=1e6, volatility=0.02
    )
    d = br.to_dict()
    assert d["total_bps"] >= 0


def test_pre_trade_cost_estimate():
    est = pre_trade_cost_estimate(
        side="buy",
        quantity=1000,
        mid=100.0,
        spread=0.02,
        adv=1e6,
        volatility=0.02,
        financing_rate=0.05,
        financing_days=1,
        borrow_rate=0.02,
        borrow_days=1,
        fx_cost_bps=0.5,
    )
    assert est["total_cost"] > 0
    assert "components" in est
    assert est["total_cost_bps"] >= 0


def test_post_trade_is_attribution():
    fills = [
        {"qty": 40, "price": 100.05},
        {"qty": 60, "price": 100.08},
    ]
    post = post_trade_cost_analysis(
        fills,
        side="buy",
        arrival_price=100.0,
        decision_price=100.0,
        mid=100.02,
        spread=0.02,
        parent_quantity=100.0,
        benchmark_vwap=100.03,
        benchmark_twap=100.04,
        financing_rate=0.01,
        financing_days=1,
    )
    assert "implementation_shortfall" in post
    assert "cost_attribution" in post
    assert post["filled_quantity"] == 100.0
    assert post["fill_rate"] == 1.0
    # Sell side
    post_s = post_trade_cost_analysis(
        [{"quantity": 50, "fill_price": 99.9}],
        side="sell",
        arrival_price=100.0,
        parent_quantity=80.0,
        mid=99.8,
    )
    assert post_s["residual_quantity"] == 30.0
    assert post_s["implementation_shortfall"] is not None


def test_component_cost_helpers():
    assert commission_cost(quantity=100, price=100, commission_bps=1.0)["total"] > 0
    assert exchange_fees(quantity=100, price=100, fee_bps=0.3)["total"] > 0
    assert spread_cost(quantity=100, mid=100, spread=0.02, side="buy")["total"] > 0
    assert slippage_cost(side="buy", quantity=100, mid=100, spread=0.02, adv=1e6)["total"] >= 0
    assert market_impact_cost(side="buy", quantity=100, mid=100, adv=1e6)["total"] >= 0
    assert financing_cost(notional=10_000, rate=0.05, days=2)["total"] > 0
    assert borrow_cost(notional=10_000, borrow_rate=0.02, days=2, is_short=True)["total"] > 0
    assert borrow_cost(notional=10_000, borrow_rate=0.02, days=2, is_short=False)["total"] == 0


def test_engine_estimate_costs_and_slippage(engine, market_context, make_limit_order, order_manager):
    order = make_limit_order(quantity=100)
    costs = engine.estimate_costs([order], market_context)
    assert costs["total_cost"] >= 0
    slip = engine.estimate_slippage([order], market_context)
    assert "orders" in slip
    # delta map form
    costs2 = engine.estimate_costs({"AAPL": 50.0}, market_context)
    assert costs2["total_cost"] >= 0
    # kwargs form
    slip2 = engine.estimate_slippage(side="buy", quantity=100, market_context=market_context)
    assert slip2["expected_slippage_bps"] >= 0
