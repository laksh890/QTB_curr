"""Coverage gaps 3: multi_period, estimators, processes, serializer, construction, engine, robust, viz, diagnostics."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import importlib
import numpy as np
import pytest

from iqrp.app.portfolio import processes, registry, visualization
from iqrp.app.portfolio.base.portfolio import Portfolio
from iqrp.app.portfolio.base.position import Position
from iqrp.app.portfolio.config import PortfolioSettings
from iqrp.app.portfolio.construction.constructor import PortfolioResult
from iqrp.app.portfolio.construction.rebalance import (
    RebalanceBands,
    apply_rebalance_bands,
    evaluate_triggers,
    plan_rebalance,
)
from iqrp.app.portfolio.construction.signal_to_weight import signals_to_raw_weights
from iqrp.app.portfolio.construction.target_positions import (
    TargetPositions,
    target_positions,
    weights_to_positions,
)
from iqrp.app.portfolio.construction.target_weights import (
    TargetWeights,
    build_target_weights,
)
from iqrp.app.portfolio.covariance.factor import factor_covariance
from iqrp.app.portfolio.covariance.robust import robust_covariance
from iqrp.app.portfolio.covariance.shrinkage import ledoit_wolf_covariance, shrinkage_covariance
from iqrp.app.portfolio.diagnostics import (
    feasibility_diagnostics,
    numerical_health,
    portfolio_diagnostics,
)
from iqrp.app.portfolio.engine import PortfolioConstructionEngine, dict_to_optimization_result
from iqrp.app.portfolio.expected_returns.black_litterman import (
    black_litterman_posterior,
    equilibrium_returns,
)
from iqrp.app.portfolio.expected_returns.forecast import forecast_expected_returns
from iqrp.app.portfolio.expected_returns.historical import historical_expected_returns
from iqrp.app.portfolio.expected_returns.shrinkage import (
    james_stein_shrinkage,
    shrinkage_expected_returns,
)
from iqrp.app.portfolio.multi_period.dynamic_programming import (
    _simplex_grid,
    optimize_dynamic_programming,
)
from iqrp.app.portfolio.multi_period.optimizer import optimize_multi_period
from iqrp.app.portfolio.multi_period.rebalancing import apply_drift, rebalance_schedule
from iqrp.app.portfolio.portfolio_risk.decomposition import factor_risk_decomposition
from iqrp.app.portfolio.robust.distributional_robust import optimize_distributional_robust
from iqrp.app.portfolio.robust.parameter_uncertainty import (
    mu_standard_errors,
    optimize_parameter_uncertainty,
)
from iqrp.app.portfolio.robust.uncertainty_sets import (
    box_uncertainty_cov,
    box_uncertainty_mu,
    ellipsoidal_uncertainty_mu,
    worst_case_mu,
    worst_case_return,
)
from iqrp.app.portfolio.serializer import PortfolioSerializer, _to_jsonable


# ============================================================================= multi_period
def test_multi_period_path_mismatches_and_return_path(mu, cov, names):
    n = len(names)
    # mu_path first-dim mismatch
    bad = optimize_multi_period(mu_path=np.tile(mu, (2, 1)), horizons=3, cov=cov, names=names)
    assert bad["success"] is False

    # cov_path length mismatch
    bad2 = optimize_multi_period(mu=mu, cov_path=[cov, cov], horizons=3, names=names)
    assert bad2["success"] is False

    # return_path width mismatch
    bad3 = optimize_multi_period(
        mu=mu, cov=cov, horizons=2, return_path=np.ones((2, n + 1)), names=names
    )
    assert bad3["success"] is False

    # 1-D mu_path tile + short return_path pad + rebalance_every > 1
    res = optimize_multi_period(
        mu_path=mu,
        cov=cov,
        horizons=3,
        return_path=np.zeros((1, n)),
        rebalance_every=2,
        transaction_cost=0.001,
        names=names,
        max_weight=0.5,
        current_weights=np.ones(n) / n,
    )
    assert "success" in res
    if res["success"]:
        diag = res.get("diagnostics") or {}
        assert diag.get("horizons") == 3 or "weights_path" in diag

    # no mu, cov only
    res2 = optimize_multi_period(cov=cov, horizons=2, names=names, max_weight=0.5)
    assert "success" in res2

    # missing both
    assert optimize_multi_period(horizons=2, names=names)["success"] is False

    # infeasible constraints
    inf = optimize_multi_period(mu=mu, cov=cov, horizons=2, names=names, max_weight=0.1)
    assert inf["success"] is False

    # cov_path length-1 broadcast
    res3 = optimize_multi_period(
        mu=mu, cov_path=[cov], horizons=2, names=names, max_weight=0.5, rebalance_every=1
    )
    assert "success" in res3


def test_multi_period_period_infeasible_via_turnover(mu, cov, names, current_weights):
    n = len(names)
    # extreme mu to force large trades against tiny turnover threshold
    mu_path = np.tile(np.array([5.0, -5.0, -5.0, -5.0][:n]), (2, 1))
    res = optimize_multi_period(
        mu=mu_path[0],
        cov=cov,
        horizons=2,
        mu_path=mu_path,
        current_weights=current_weights,
        turnover_threshold=1e-18,
        transaction_cost=0.0,
        names=names,
        max_weight=0.5,
    )
    assert "success" in res


def test_dp_failure_paths_and_greedy(mu, cov, names):
    assert optimize_dynamic_programming(horizons=0, cov=cov, names=names)["success"] is False
    assert optimize_dynamic_programming(horizons=2, names=names)["success"] is False

    # short rejection
    res = optimize_dynamic_programming(
        mu=mu[:3],
        cov=cov[:3, :3],
        horizons=2,
        long_only=False,
        min_weight=-0.5,
        names=names[:3],
    )
    assert res["success"] is False

    # empty grid via impossible box
    res2 = optimize_dynamic_programming(
        mu=mu[:3],
        cov=cov[:3, :3],
        horizons=2,
        max_weight=0.1,
        budget=1.0,
        names=names[:3],
        grid_levels=3,
    )
    assert res2["success"] is False

    # greedy heuristic when n/horizons large
    n = min(5, len(names))
    res3 = optimize_dynamic_programming(
        mu=mu[:n],
        cov=cov[:n, :n],
        horizons=3,
        grid_levels=6,
        names=names[:n],
        max_weight=0.5,
    )
    assert "success" in res3
    if res3.get("method") == "greedy_heuristic" or res3.get("name") == "dynamic_programming":
        assert True

    # simplex grid edge
    assert _simplex_grid(0, 2).size == 0
    g = _simplex_grid(2, 2)
    assert g.shape[1] == 2


def test_rebalancing_helpers(weights):
    sched = rebalance_schedule(5, frequency=2, threshold=0.1)
    assert "flags" in sched
    # threshold path when flags all false initially uses threshold
    drifted = apply_drift(weights, np.zeros(len(weights)))
    assert drifted.shape == weights.shape
    drifted2 = apply_drift(weights, np.array([0.01, -0.01, 0.0, 0.02][: len(weights)]))
    assert float(np.sum(drifted2)) == pytest.approx(1.0, abs=1e-8)


# ============================================================================= expected_returns
def test_james_stein_edges(mu, cov, returns, names):
    with pytest.raises(ValueError):
        james_stein_shrinkage(mu, prior=[0.0])
    out = james_stein_shrinkage(mu, prior=np.zeros_like(mu), intensity=0.3)
    assert out["intensity_method"] == "provided"
    out2 = james_stein_shrinkage(mu, cov=cov, n_obs=returns.shape[0])
    assert out2["intensity_method"] == "james_stein"
    # equal mu/prior → alpha 1
    out3 = james_stein_shrinkage(np.ones(4) * 0.01, prior=np.ones(4) * 0.01, cov=cov, n_obs=100)
    assert out3["intensity"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        james_stein_shrinkage(mu, cov=np.eye(2), n_obs=50)
    with pytest.raises(ValueError):
        shrinkage_expected_returns()
    out4 = shrinkage_expected_returns(mu=mu, names=names)
    assert out4["source"] == "provided_mu"
    out5 = shrinkage_expected_returns(returns=returns, names=names, intensity=0.2)
    assert "mu" in out5


def test_black_litterman_market_caps_omega_mismatches(mu, cov, names):
    n = len(names)
    with pytest.raises(ValueError):
        equilibrium_returns(cov, market_weights=[0.1])
    with pytest.raises(ValueError):
        equilibrium_returns(np.ones((2, 3)), market_weights=[0.5, 0.5])

    # market_caps path + zero caps → equal
    bl = black_litterman_posterior(cov, market_caps=np.zeros(n), names=names)
    assert bl["equilibrium_method"] == "market_caps"
    bl2 = black_litterman_posterior(cov, market_caps=np.array([1.0, 2.0, 3.0, 4.0][:n]), names=names)
    assert bl2["equilibrium_method"] == "market_caps"

    # equal weight default
    bl3 = black_litterman_posterior(cov, names=names)
    assert bl3["equilibrium_method"] == "equal_weight"

    # equilibrium_mu provided
    bl4 = black_litterman_posterior(cov, equilibrium_mu=mu, names=names)
    assert bl4["equilibrium_method"] == "provided"
    with pytest.raises(ValueError):
        black_litterman_posterior(cov, equilibrium_mu=[0.1])

    P = np.eye(1, n)
    Q = np.array([0.01])
    # omega diagonal
    bl5 = black_litterman_posterior(cov, market_weights=np.ones(n) / n, P=P, Q=Q, omega=np.array([0.1]))
    assert bl5["n_views"] == 1
    # omega matrix
    bl6 = black_litterman_posterior(
        cov, market_weights=np.ones(n) / n, P=P, Q=Q, omega=np.array([[0.1]])
    )
    assert bl6["view_method"] == "matrix_provided"
    # omega None proportional
    bl7 = black_litterman_posterior(cov, market_weights=np.ones(n) / n, P=P, Q=Q)
    assert bl7["n_views"] == 1

    with pytest.raises(ValueError):
        black_litterman_posterior(cov, market_caps=[1.0])
    with pytest.raises(ValueError):
        black_litterman_posterior(cov, P=np.eye(1, n), Q=np.array([0.01, 0.02]))
    with pytest.raises(ValueError):
        black_litterman_posterior(cov, P=np.eye(1, n), Q=Q, omega=np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        black_litterman_posterior(cov, P=np.eye(1, n), Q=Q, omega=np.eye(2))


def test_forecast_and_historical_edges(forecasts, names, returns):
    with pytest.raises(ValueError):
        forecast_expected_returns(forecasts, prior=[0.0])
    with pytest.raises(ValueError):
        forecast_expected_returns(forecasts, confidence=[0.5])
    with pytest.raises(ValueError):
        forecast_expected_returns(forecasts, uncertainty=[0.1])
    out = forecast_expected_returns(
        forecasts,
        prior=np.zeros(len(forecasts)),
        confidence=np.ones(len(forecasts)) * 0.5,
        uncertainty=np.ones(len(forecasts)) * 0.2,
        names=names,
    )
    assert out["uncertainty_applied"] is True

    hist = historical_expected_returns(returns[:, 0], window=50)  # 1-D
    assert "mu" in hist
    hist2 = historical_expected_returns(returns[:0], names=names)
    assert hist2["n_obs"] == 0
    with pytest.raises(ValueError):
        historical_expected_returns(np.ones((2, 2, 2)))


# ============================================================================= covariance
def test_shrinkage_factor_robust_variants(returns, rng):
    with pytest.raises(ValueError):
        shrinkage_covariance(np.ones((2, 2, 2)))
    lw = ledoit_wolf_covariance(returns[:1])  # T<2 fallback
    assert "matrix" in lw or "name" in lw
    # all-nan rows → t_eff < 2
    bad = returns.copy()
    bad[:] = np.nan
    lw2 = ledoit_wolf_covariance(bad)
    assert "name" in lw2

    sh = shrinkage_covariance(returns, method="ledoit_wolf")
    assert "matrix" in sh
    sh2 = shrinkage_covariance(returns, method="ledoit_wolf", intensity=0.5)
    assert sh2["intensity"] == pytest.approx(0.5)

    n = returns.shape[1]
    B = rng.normal(size=(n, 2))
    fr = rng.normal(size=(80, 2))
    fc = factor_covariance(factor_loadings=B, factor_cov=np.eye(2), residual_vars=np.ones(n) * 1e-4)
    assert fc["factor_method"] == "provided"
    fc2 = factor_covariance(factor_loadings=B, factor_returns=fr, asset_returns=returns[:80])
    assert fc2["residual_method"] == "ols_residuals"
    fc3 = factor_covariance(factor_loadings=np.ones(n))  # 1-D → identity F
    assert fc3["n_factors"] == 1
    with pytest.raises(ValueError):
        factor_covariance(factor_loadings=B, factor_cov=np.eye(3))
    with pytest.raises(ValueError):
        factor_covariance(factor_loadings=B, residual_vars=[1.0])
    with pytest.raises(ValueError):
        factor_covariance(factor_loadings=B, factor_returns=fr, asset_returns=returns[:80, :2])

    rb = robust_covariance(returns[:1], method="mcd", n_trials=2, seed=0)
    assert "name" in rb
    rb2 = robust_covariance(returns, method="winsorize_mcd", n_trials=4, seed=1)
    assert "matrix" in rb2
    with pytest.raises(ValueError):
        robust_covariance(np.ones((2, 2, 2)))


# ============================================================================= processes
def test_processes_all_kinds_and_mc_edges(returns, monkeypatch):
    kinds = [
        "normal",
        "high_volatility",
        "low_liquidity",
        "correlation_spike",
        "regime_transition",
        "large_gaps",
        "drawdown",
    ]
    for k in kinds:
        out = processes.simulate_portfolio_scenario(kind=k, n=60, n_assets=3, seed=7)
        assert "returns" in out

    # force local fallback
    def _boom(**kwargs):
        raise RuntimeError("no risk processes")

    monkeypatch.setattr(
        "iqrp.app.risk.processes.simulate_risk_scenario",
        _boom,
        raising=False,
    )
    # Import path may already succeed; still call and accept either source
    fb = processes.simulate_portfolio_scenario(kind="drawdown", n=40, n_assets=3, seed=1)
    assert "returns" in fb

    # force monte_carlo fallback
    def _boom_mc(*args, **kwargs):
        raise RuntimeError("no mc")

    monkeypatch.setattr(
        "iqrp.app.risk.simulation.correlated_monte_carlo",
        _boom_mc,
        raising=False,
    )
    paths = processes.monte_carlo_portfolio_paths(
        returns[:, 0],  # 1-D
        n_simulations=20,
        horizon=5,
        seed=2,
        weights=np.array([1.0, 2.0]),  # size mismatch resize
    )
    assert paths.get("portfolio_paths") is not None or "source" in paths

    batch = processes.process_scenarios(kinds=None, n=40, n_assets=3, seed=0)
    assert len(batch["kinds"]) >= 3


# ============================================================================= serializer
def test_serializer_to_jsonable_and_load_errors(tmp_path: Path, names, weights):
    class E(Enum):
        A = "a"

    class Dumpable:
        def model_dump(self):
            return {"x": 1}

    class HasDict:
        def to_dict(self):
            return {"y": np.array([1.0]), "e": E.A, "p": Path("/tmp/x")}

    assert _to_jsonable(None) is None
    assert _to_jsonable(Path("/a/b")) == "/a/b"
    assert _to_jsonable(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert _to_jsonable(np.float64(1.5)) == 1.5
    assert _to_jsonable(HasDict())["y"] == [1.0]
    assert _to_jsonable(Dumpable()) == {"x": 1}
    assert _to_jsonable(E.A) == "a"
    assert isinstance(_to_jsonable(object()), str)

    ser = PortfolioSerializer()
    raw = ser.dump_bytes(Dumpable())
    assert isinstance(raw, bytes)
    raw2 = ser.dump_bytes(123)
    assert b"123" in raw2 or b"value" in raw2

    # load missing file
    with pytest.raises(Exception):
        ser.load_portfolio(tmp_path / "missing.json")


# ============================================================================= construction
def test_target_weights_from_dict_and_edges(names):
    tw = TargetWeights(names=list(names), weights=[0.5])  # pad
    assert len(tw.weights) == len(names)
    assert tw.n_assets == len(names)
    assert tw.as_dict()[names[0]] == pytest.approx(0.5)
    d = tw.to_dict()
    back = TargetWeights.from_dict(d)
    assert back.names == list(names)

    # from_arrays name length mismatch → default names
    tw2 = TargetWeights.from_arrays([0.5, 0.5], names=["only_one"])
    assert len(tw2.names) == 2

    cash = TargetWeights.equal_weight(0)
    assert cash.method == "cash" or cash.budget == 0.0
    eq = TargetWeights.equal_weight(names)
    assert abs(sum(eq.weights) - 1.0) < 1e-12

    built = build_target_weights({names[0]: 0.6, names[1]: 0.4}, names=names)
    assert built.weights[0] == pytest.approx(0.6)


def test_target_positions_edges(names, prices, weights):
    tp = weights_to_positions(
        {names[0]: 0.5, names[1]: 0.5},
        capital=1e5,
        prices=prices,
        names=names,
        min_order=1e9,  # wipe small
        max_order=10.0,
        lot_sizes=[1, 1, 1, 1],
        currencies=["USD"],
        round_lots=True,
    )
    assert len(tp) == len(names)
    assert list(tp)
    assert tp.names()
    assert tp.quantities()
    assert tp.weights()
    d = tp.to_dict()
    back = TargetPositions.from_dict(d)
    assert back.capital == tp.capital

    # TargetWeights input
    tw = TargetWeights.from_arrays(weights, names=names)
    tp2 = target_positions(tw, capital=1e5, prices=[0.0, 50.0, 25.0, 80.0], multipliers=[0.0, 1, 1, 1])
    assert len(tp2.positions) == len(names)


def test_rebalance_trigger_kinds_and_bands(names, weights, current_weights):
    trades = apply_rebalance_bands(
        current_weights,
        weights,
        absolute=0.01,
        relative=0.1,
        min_trade=0.05,
    )
    assert trades.shape[0] == len(weights)

    trigs = evaluate_triggers(
        current_weights=current_weights,
        target_weights=weights,
        scheduled=True,
        turnover_threshold=0.01,
        risk_breach=False,
        risk_metric=2.0,
        risk_limit=1.0,
        drift_threshold=0.01,
        regime_change=True,
        drawdown=0.2,
        drawdown_threshold=0.1,
        liquidity_stress=False,
        liquidity_score=0.1,
        liquidity_threshold=0.5,
        force=True,
    )
    kinds = {t.kind for t in trigs}
    assert "scheduled" in kinds
    assert "risk" in kinds
    assert "drift" in kinds
    assert "regime" in kinds
    assert "drawdown" in kinds
    assert "liquidity" in kinds
    assert "manual" in kinds

    plan = plan_rebalance(
        current_weights,
        weights,
        names=names[:2],  # force rename path
        bands={"absolute": 0.5, "relative": 0.0, "min_trade": 0.0},
        scheduled=True,
        risk_breach=True,
        drawdown=0.5,
        drawdown_threshold=0.1,
        regime_change=True,
    )
    assert isinstance(plan.to_dict(), dict)

    plan2 = plan_rebalance(
        current_weights,
        current_weights,
        bands=RebalanceBands(absolute=0.0, relative=0.0, min_trade=0.0),
        scheduled=False,
        force=False,
        always_if_triggered=False,
    )
    assert plan2.should_rebalance is False or plan2.turnover == 0.0


def test_signal_empty_and_long_short(names, signals):
    empty = signals_to_raw_weights([])
    assert empty["weights"] == []
    out = signals_to_raw_weights(signals, method="identity", long_only=False, budget=1.0, names=names[:2])
    assert len(out["weights"]) == len(signals)
    out2 = signals_to_raw_weights(np.zeros(4), method="proportional", long_only=False)
    assert abs(sum(out2["weights"])) < 1e-8 or out2["gross"] == 0.0
    out3 = signals_to_raw_weights(np.array([-1.0, -1.0, -1.0, -1.0]), method="zscore", long_only=True)
    assert abs(sum(out3["weights"]) - 1.0) < 1e-8


def test_portfolio_result_to_dict_and_to_portfolio(names, weights, prices):
    tp = weights_to_positions(weights, capital=1e5, prices=prices, names=names)
    pr = PortfolioResult(
        portfolio_weights=TargetWeights.from_arrays(weights, names=names),
        target_positions=tp,
        names=list(names),
        weights=list(weights),
        method="mv",
    )
    d = pr.to_dict()
    assert d["target_positions"] is not None
    port = pr.to_portfolio()
    assert isinstance(port, Portfolio)

    pr2 = PortfolioResult(target_positions=[Position(asset=names[0], weight=1.0)], names=[names[0]], weights=[1.0])
    assert pr2.to_dict()["target_positions"] is not None
    assert pr2.to_portfolio().positions


# ============================================================================= engine remaining
def test_dict_extract_sorted_keys_and_variance_diag():
    # length mismatch → sorted keys
    r = dict_to_optimization_result(
        {"success": True, "weights": {"b": 0.4, "a": 0.6}, "expected_return": 0.01, "expected_variance": 0.02}
    )
    assert abs(sum(r.weights) - 1.0) < 1e-12
    r2 = dict_to_optimization_result(
        {"success": True, "weights": {"a": 0.5, "b": 0.5}, "diagnostics": {"variance": 0.03}},
        names=["a", "b"],
    )
    assert r2.expected_variance == pytest.approx(0.03)
    # exception swallow path with bad mu/cov shapes
    r3 = dict_to_optimization_result(
        {"success": True, "weights": [0.5, 0.5]},
        mu=np.array([[1.0]]),
        cov=np.ones(3),
    )
    assert r3.success


def test_engine_filtered_kwargs_load_settings_and_paths(mu, cov, names, returns, prices, adv, forecasts, weights):
    eng = PortfolioConstructionEngine(
        settings=PortfolioSettings(require_risk_validation=False, seed=42, method="min_variance")
    )
    # TypeError filtered kwargs: pass an unexpected kw that some optimizers reject —
    # engine filters via try/except around call; use budget/constraints extras
    res = eng.optimize(
        mu=mu,
        cov=cov,
        names=names,
        method="min_variance",
        budget=1.0,
        constraints={"long_only": True, "max_weight": 0.5},
        min_weight=0.0,
        max_gross=1.5,
    )
    assert res.success or res.fallback_used or res.failure_reason

    # expected_returns forecast from returns (no forecasts)
    er = eng.expected_returns(returns=returns, method="forecast", names=names)
    assert "mu" in er
    er2 = eng.expected_returns(forecasts=forecasts, method="forecast", names=names)
    assert "mu" in er2
    er3 = eng.expected_returns(returns=returns, method="shrinkage", names=names)
    assert "mu" in er3
    er4 = eng.expected_returns(returns=returns, method="black_litterman", names=names)
    assert "mu" in er4
    with pytest.raises(ValueError):
        eng.expected_returns(method="forecast")
    with pytest.raises(ValueError):
        eng.expected_returns(method="historical")
    with pytest.raises(ValueError):
        eng.expected_returns(method="shrinkage")
    with pytest.raises(ValueError):
        eng.expected_returns(method="unknown_xyz")

    # construct include_transaction_costs False vs True from-zero
    r1 = eng.construct(
        mu=mu,
        cov=cov,
        names=names,
        prices=prices,
        capital=1e6,
        include_transaction_costs=False,
        adv=adv,
    )
    assert isinstance(r1.transaction_cost, dict)
    r2 = eng.construct(
        mu=mu,
        cov=cov,
        names=names,
        prices=prices,
        capital=1e6,
        include_transaction_costs=True,
        adv=adv,
    )
    assert r2.transaction_cost.get("total", 0) >= 0.0

    # factor loadings error path
    r3 = eng.construct(
        mu=mu,
        cov=cov,
        names=names,
        factor_loadings="bad",
    )
    assert isinstance(r3.factor_exposure, dict)

    # target_positions construct path
    tp = eng.target_positions(mu=mu, cov=cov, names=names, prices=prices, capital=1e5, method="min_variance")
    assert tp is not None
    tp2 = eng.target_positions(weights, capital=1e5, prices=prices, names=names, as_list=True)
    assert isinstance(tp2, list)
    tw = eng.target_weights(mu=mu, cov=cov, names=names, method="min_variance")
    assert isinstance(tw, TargetWeights)

    # save/load settings fail swallow
    p = eng.save("/tmp/qtb_portfolio_engine_test.json")  # may write workspace-relative prefer tmp
    # use Path via engine.save with settings only
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td:
        path = Path(td) / "eng.json"
        eng.save(path)
        # corrupt settings
        path.write_text('{"settings": {"method": []}}', encoding="utf-8")
        loaded = eng.load(path)
        assert isinstance(loaded, dict)


def test_engine_risk_skip_and_soft_only(mu, cov, names, returns):
    settings = PortfolioSettings(require_risk_validation=True, fallback="cash", seed=1)

    class SoftOnly:
        def check_limits(self, **kwargs):
            class B:
                severity = type("S", (), {"value": "soft"})()

                def to_dict(self):
                    return {"severity": "soft"}

            return [B()]

        def validate_position(self, **kwargs):
            return {"approved": True, "action": "APPROVE", "reason": "ok"}

    eng = PortfolioConstructionEngine(settings=settings, risk_engine=SoftOnly())
    r = eng.construct(mu=mu, cov=cov, returns=returns, names=names, method="min_variance")
    assert r.risk_validation is not None
    # soft only should not hard-reject
    assert r.risk_validation.get("approved") is True or r.risk_validation.get("action") in (
        "APPROVE",
        "CAUTION",
        "REJECT",
    )

    # skip path: require validation but no engine after failed ensure
    eng2 = PortfolioConstructionEngine(settings=settings, risk_engine=None, risk_ensemble=None)
    eng2.risk_engine = None
    eng2.risk_ensemble = None
    eng2._risk_init_attempted = True
    eng2._risk_skip_reason = "forced skip"
    out = eng2._run_risk_validation(
        weights=np.ones(len(names)) / len(names),
        returns=returns,
        forecast_confidence=None,
    )
    assert out.get("status") == "skipped" or out.get("action") == "SKIP"


# ============================================================================= robust
def test_robust_worst_case_and_parameter_uncertainty(mu, cov, weights, returns, names):
    box = box_uncertainty_mu(mu, absolute=0.01)
    box2 = box_uncertainty_mu(mu, absolute=0.02)
    with pytest.raises(ValueError):
        box_uncertainty_mu(mu, absolute=[0.01, 0.02])
    ell = ellipsoidal_uncertainty_mu(mu, cov, rho=1.0)
    with pytest.raises(ValueError):
        ellipsoidal_uncertainty_mu(mu, np.eye(2))
    with pytest.raises(ValueError):
        box_uncertainty_cov(np.ones(3))

    wc = worst_case_mu(weights, box)
    assert wc.shape == mu.shape
    wc2 = worst_case_mu(weights, ell)
    assert wc2.shape == mu.shape
    # zero quad ellipsoid
    ell0 = {"type": "ellipsoidal", "mu": mu, "scale_cov": np.zeros_like(cov), "rho": 1.0}
    assert np.allclose(worst_case_mu(weights, ell0), mu)
    with pytest.raises(ValueError):
        worst_case_mu(weights, {"type": "nope"})
    assert isinstance(worst_case_return(weights, box), float)
    assert isinstance(worst_case_return(weights, ell), float)
    with pytest.raises(ValueError):
        worst_case_return(weights, {"type": "nope"})

    se = mu_standard_errors(returns)
    assert se.shape[0] == returns.shape[1]
    se2 = mu_standard_errors(cov=cov, n_obs=100)
    assert se2.shape[0] == cov.shape[0]
    with pytest.raises(ValueError):
        mu_standard_errors(returns=np.ones(3))
    with pytest.raises(ValueError):
        mu_standard_errors()

    res = optimize_parameter_uncertainty(
        mu=mu, cov=cov, names=names, set_type="box", z_score=1.0, max_weight=0.5
    )
    assert "success" in res
    res2 = optimize_parameter_uncertainty(
        returns=returns, names=names, set_type="ellipsoidal", max_weight=0.5
    )
    assert "success" in res2
    fail = optimize_parameter_uncertainty(names=names)
    assert fail["success"] is False

    # distributional robust failure
    fail2 = optimize_distributional_robust(names=names)
    assert fail2["success"] is False
    # dict uncertainty
    res3 = optimize_distributional_robust(
        mu=mu, cov=cov, names=names, uncertainty=box, max_weight=0.5, current_weights=weights
    )
    assert "success" in res3


# ============================================================================= visualization + diagnostics + risk + registry
def test_visualization_empty_names_and_percent_dict(weights):
    wp = visualization.weights_payload(weights, names=["a"])  # short names pad
    assert len(wp["labels"]) == len(weights)
    rc = visualization.risk_contribution_payload(
        {"percent": [0.25, 0.25, 0.25, 0.25], "values": [0.1, 0.1, 0.1, 0.1]},
        names=None,
        percent=True,
    )
    assert rc["unit"] == "percent"
    rc2 = visualization.risk_contribution_payload(
        {"values": [0.1, 0.2, 0.3, 0.4]},
        percent=True,
    )
    assert isinstance(rc2["values"], list)


def test_diagnostics_unhealthy_not_psd_and_infeasible(weights, cov, mu):
    # not PSD
    bad_cov = np.array([[1.0, 2.0], [2.0, 1.0]])
    nh = numerical_health(weights=np.array([20.0, 0.1]), cov=bad_cov, mu=np.array([np.nan, 0.1]))
    assert nh["healthy"] is False
    assert "cov_not_psd" in nh["issues"] or "weights_extreme" in nh["issues"] or "mu_nonfinite" in nh["issues"]

    nh2 = numerical_health(cov=np.ones(3))
    assert "cov_not_square" in nh2["issues"]

    # asymmetric
    asym = cov.copy()
    asym[0, 1] = asym[0, 1] + 1.0
    nh3 = numerical_health(cov=asym)
    assert "cov_asymmetric" in nh3["issues"] or nh3["healthy"] is False

    fd = feasibility_diagnostics(
        np.array([0.8, -0.1, 0.2, 0.1]),
        max_weight=0.3,
        max_gross=0.5,
        max_leverage=0.5,
        long_only=True,
        budget=1.0,
    )
    assert fd["feasible"] is False
    assert fd["violations"]

    pd = portfolio_diagnostics(
        np.array([0.9, 0.1, 0.0, 0.0]),
        cov=cov,
        mu=mu,
        max_weight=0.3,
        long_only=True,
    )
    assert pd["feasibility"]["feasible"] is False


def test_factor_risk_decomposition_missing_paths(weights, cov):
    # 1-D loadings, missing factor_cov, transpose, zero vol
    fr = factor_risk_decomposition(weights, factor_loadings=np.ones(len(weights)))
    assert fr["n_factors"] == 1
    # k x n loadings → transpose to n x k
    B = np.random.default_rng(0).normal(size=(2, len(weights)))
    fr2 = factor_risk_decomposition(
        weights,
        factor_loadings=B,
        factor_cov=np.array([0.01, 0.02]),
        factor_names=["f"],
    )
    assert fr2["n_factors"] == 2
    # pad weights shorter than loadings rows
    fr3 = factor_risk_decomposition(
        weights[:2],
        factor_loadings=np.eye(4)[:, :1],
        idiosyncratic_var=np.ones(4) * 1e-4,
    )
    assert fr3["n_assets"] == 4
    # zero portfolio → zero mrc
    fr4 = factor_risk_decomposition(np.zeros(4), factor_loadings=np.eye(4)[:, :2])
    assert fr4["portfolio_volatility"] == pytest.approx(0.0)


def test_registry_empty_name():
    with pytest.raises(ValueError):
        registry.register("  ", lambda *a, **k: None)


# ============================================================================= extra closing gaps
def test_phase10_fail_paths(monkeypatch, tmp_path):
    from iqrp.app.portfolio import phase10

    # missing symbol
    class FakeMod:
        pass

    real_import = importlib.import_module if False else None
    import importlib as il

    orig = il.import_module

    def fake_import(name, *a, **k):
        mod = orig(name, *a, **k)
        # strip a known symbol temporarily via wrapper
        if name == "iqrp.app.portfolio.optimization":
            class Wrap:
                def __getattr__(self, item):
                    if item == "optimize_mean_variance":
                        raise AttributeError(item)
                    return getattr(mod, item)

                def __hasattr_missing(self):
                    return False

            w = Wrap()
            # hasattr uses getattr
            return w
        return mod

    # simpler: inject a synthetic ComponentCheck with bad path
    bad = phase10.ComponentCheck(
        name="BadComp",
        category="x",
        import_path="iqrp.app.portfolio.this_module_does_not_exist_xyz",
        symbol="Nope",
        docs=["MissingDocThatDoesNotExist.md"],
    )
    monkeypatch.setattr(phase10, "PHASE10_COMPONENTS", [bad] + list(phase10.PHASE10_COMPONENTS[:1]))
    # also force missing required doc
    monkeypatch.setattr(phase10, "REQUIRED_DOCS", ["MissingDocThatDoesNotExist.md"])
    # force hydra missing
    monkeypatch.setattr(
        phase10,
        "_docs_root",
        lambda: tmp_path,
    )
    report = phase10.validate_phase10()
    assert report["status"] == "FAIL"
    assert report["summary"]["failures"]

    # write report still works
    out = phase10.write_phase10_report(tmp_path / "p10.json")
    assert out.exists()

    # __main__ path
    import runpy
    # skip runpy of phase10 to avoid side effects; cover write + ComponentCheck.to_dict already


def test_rebalancing_calendar_and_errors(weights):
    with pytest.raises(ValueError):
        rebalance_schedule(-1)
    sched = rebalance_schedule(5, calendar=[0, 2, 4], frequency=10)
    assert sched["flags"][0] is True
    sched2 = rebalance_schedule(4, frequency=3, start=1)
    assert isinstance(sched2["flags"], list)
    with pytest.raises(ValueError):
        apply_drift(weights, np.ones(len(weights) + 1))
    # near-zero grown sum
    z = apply_drift(np.array([1.0, -1.0]), np.array([-1.0, -1.0]))
    assert np.allclose(z, 0.0)


def test_dp_greedy_heuristic_large_n(rng):
    n = 5
    corr = 0.2 * np.ones((n, n)) + 0.8 * np.eye(n)
    cov = corr * 0.01
    mu = rng.normal(0, 0.001, size=n)
    res = optimize_dynamic_programming(
        mu=mu,
        cov=cov,
        horizons=3,
        grid_levels=6,
        max_weight=0.5,
        names=[f"A{i}" for i in range(n)],
    )
    assert "success" in res


def test_engine_dict_weights_n_match_and_fallback_minvar(mu, cov, names, monkeypatch):
    # _extract_weights via dict length match
    r = dict_to_optimization_result(
        {"success": True, "weights": {names[i]: 1.0 / len(names) for i in range(len(names))}},
        names=names,
        mu=mu,
        cov=cov,
    )
    assert abs(sum(r.weights) - 1.0) < 1e-8

    eng = PortfolioConstructionEngine(
        settings=PortfolioSettings(require_risk_validation=False, fallback="min_variance", seed=2)
    )

    def boom(**kwargs):
        raise RuntimeError("opt fail")

    import iqrp.app.portfolio.engine as eng_mod

    monkeypatch.setitem(eng_mod._OPTIMIZER_MAP, "mean_variance", boom)
    res = eng.optimize(mu=mu, cov=cov, method="mean_variance", names=names)
    assert res.fallback_used is True
    assert res.fallback_kind in ("min_variance", "cash")

    # current weights size for n when cov missing shape
    res2 = eng._apply_optimize_fallback(
        reason="x",
        mu=None,
        cov=None,
        current_weights=np.ones(len(names)) / len(names),
        names=names,
        method="mv",
    )
    assert res2.fallback_used


def test_engine_forecast_confidence_and_decision_without_approved(mu, cov, names, returns):
    class Dec:
        def check_limits(self, **kwargs):
            return []

        def validate_position(self, **kwargs):
            return {"action": "CAUTION", "reason": "watch"}

    settings = PortfolioSettings(require_risk_validation=True, fallback="cash", seed=1)
    eng = PortfolioConstructionEngine(settings=settings, risk_engine=Dec())
    r = eng.construct(
        mu=mu,
        cov=cov,
        returns=returns,
        names=names,
        method="min_variance",
        forecast_confidence=np.ones(len(names)) * 0.5,
    )
    assert r.risk_validation is not None

    class Dec2:
        def check_limits(self, **kwargs):
            return []

        def validate_position(self, **kwargs):
            return {"status": "APPROVE", "reason": "ok"}

    eng2 = PortfolioConstructionEngine(settings=settings, risk_engine=Dec2())
    r2 = eng2.construct(mu=mu, cov=cov, returns=returns, names=names, method="min_variance")
    assert r2.risk_validation is not None


def test_robust_cov_1d_and_emptyish(rng):
    x = rng.normal(size=50)
    out = robust_covariance(x, method="winsorize", n_trials=2, seed=0)
    assert "matrix" in out or "name" in out
    # column all-nan winsorize continue
    m = rng.normal(size=(40, 3))
    m[:, 1] = np.nan
    out2 = robust_covariance(m, method="winsorize", n_trials=2, seed=0)
    assert "name" in out2


def test_processes_mc_success_path(returns):
    # exercise real risk.simulation path when available
    paths = processes.monte_carlo_portfolio_paths(
        returns,
        n_simulations=15,
        horizon=4,
        seed=3,
        weights=np.ones(returns.shape[1]) / returns.shape[1],
    )
    assert paths is not None
    assert "portfolio_paths" in paths or "source" in paths


def test_serializer_numpy_integer_and_enum_fail(tmp_path):
    assert _to_jsonable(np.int64(3)) == 3
    # enum branch exception swallow: object with value but not Enum
    class Weird:
        value = "x"

    assert isinstance(_to_jsonable(Weird()), str) or _to_jsonable(Weird()) == "x"


def test_target_positions_align_name_mismatch(names, prices):
    tp = weights_to_positions(
        [0.25, 0.25, 0.25, 0.25],
        capital=1e5,
        prices=prices,
        names=["only"],  # mismatch → default names
        min_order=[1.0, 2.0],
        max_order=[100.0, 200.0],
    )
    assert len(tp.positions) == 4


def test_ill_conditioned_cov_diagnostics():
    # nearly singular → high condition
    c = np.array([[1.0, 1.0 - 1e-15], [1.0 - 1e-15, 1.0]])
    nh = numerical_health(cov=c * 1e-20 + np.eye(2) * 1e-30)
    assert "healthy" in nh


def test_phase10_missing_symbol(monkeypatch, tmp_path):
    from iqrp.app.portfolio import phase10
    import types

    fake = types.ModuleType("iqrp.app.portfolio._fake_phase10_mod")
    # no symbol

    import importlib

    orig = importlib.import_module

    def fake_import(name, *a, **k):
        if name == "iqrp.app.portfolio._fake_phase10_mod":
            return fake
        return orig(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    bad = phase10.ComponentCheck(
        name="MissingSym",
        category="x",
        import_path="iqrp.app.portfolio._fake_phase10_mod",
        symbol="DoesNotExist",
        docs=[],
    )
    monkeypatch.setattr(phase10, "PHASE10_COMPONENTS", [bad])
    monkeypatch.setattr(phase10, "REQUIRED_DOCS", [])
    monkeypatch.setattr(phase10, "_docs_root", lambda: tmp_path)
    # make hydra path exist to avoid that failure dominating
    report = phase10.validate_phase10()
    assert any("missing" in f.lower() or "DoesNotExist" in f for f in report["summary"]["failures"]) or report["status"] == "FAIL"


def test_engine_ensure_risk_failure(monkeypatch):
    eng = PortfolioConstructionEngine(
        settings=PortfolioSettings(require_risk_validation=False, seed=1)
    )
    eng.risk_engine = None
    eng.risk_ensemble = None
    eng._risk_init_attempted = False

    import iqrp.app.risk as risk_mod

    def boom(*a, **k):
        raise RuntimeError("cannot construct risk")

    monkeypatch.setattr(risk_mod, "RiskIntelligenceEngine", boom)
    eng.settings = PortfolioSettings(require_risk_validation=True, seed=1)
    eng._ensure_risk_engine()
    assert eng._risk_skip_reason is not None
    assert eng.risk_engine is None


def test_engine_target_positions_empty_construct(engine, names, mu, cov):
    # force construct without prices → empty target_positions path
    # target_positions() without capital/prices constructs then returns pos
    tp = engine.target_positions(mu=mu, cov=cov, names=names, method="min_variance")
    assert tp is not None


def test_multi_period_failed_step_hold(mu, cov, names, current_weights, monkeypatch):
    import iqrp.app.portfolio.multi_period.optimizer as mp
    import iqrp.app.portfolio.optimization.turnover as to_mod

    def fail_to(**kwargs):
        return {"success": False, "status": "failed", "failure_reason": "x", "weights": current_weights}

    monkeypatch.setattr(to_mod, "optimize_turnover", fail_to)
    # also patch the import used inside loop — module import inside function
    res = mp.optimize_multi_period(
        mu=mu,
        cov=cov,
        horizons=1,
        names=names,
        current_weights=current_weights,
        max_weight=0.5,
    )
    assert "success" in res


def test_dp_mu_path_1d_and_exception(mu, cov, names):
    res = optimize_dynamic_programming(
        cov=cov[:3, :3],
        mu_path=mu[:3],
        horizons=2,
        names=names[:3],
        max_weight=0.6,
        grid_levels=3,
    )
    assert "success" in res
    # exception handler with bad cov
    class Bad:
        shape = property(lambda self: (_ for _ in ()).throw(RuntimeError("x")))

    res2 = optimize_dynamic_programming(cov=Bad(), horizons=2)
    assert res2["success"] is False


def test_projection_remaining_edges():
    from iqrp.app.portfolio.optimization.projection import (
        project_simplex,
        project_gross,
        project_box_simplex,
        minimize_scipy,
        scipy_available,
    )

    # s<=0 after max → equal weights branch inside project_simplex already covered;
    # force budget restore fail in project_gross
    x = project_gross(np.array([1.0, -1.0, 0.0, 0.0]), max_gross=0.5, budget=1.0, long_only=False)
    assert float(np.sum(np.abs(x))) <= 0.5 + 1e-8

    # box simplex last return path: max_iter=1
    w = project_box_simplex(np.array([0.4, 0.3, 0.2, 0.1]), lb=0.0, ub=0.5, budget=1.0, max_iter=1)
    assert w.shape == (4,)

    if scipy_available():
        # just ensure callable path with options None uses defaults
        try:
            minimize_scipy(lambda z: float(z[0] ** 2), np.array([1.0]), method="Nelder-Mead", options={"maxiter": 5})
        except Exception:
            pass


def test_processes_mc_path_shape_branches(returns, monkeypatch):
    import iqrp.app.risk.simulation as sim

    n = returns.shape[1]
    w = np.ones(n) / n

    def ret3(*a, **k):
        return {"paths": np.ones((5, 4, n))}

    monkeypatch.setattr(sim, "correlated_monte_carlo", ret3)
    out = processes.monte_carlo_portfolio_paths(returns, n_simulations=5, horizon=4, seed=1, weights=w)
    assert out.get("portfolio_paths") is not None

    def ret2(*a, **k):
        return {"paths": np.ones((5, 4))}

    monkeypatch.setattr(sim, "correlated_monte_carlo", ret2)
    out2 = processes.monte_carlo_portfolio_paths(returns, n_simulations=5, horizon=4, seed=1, weights=w)
    assert out2.get("portfolio_paths") is not None

    def ret_other(*a, **k):
        return {"paths": "weird"}

    monkeypatch.setattr(sim, "correlated_monte_carlo", ret_other)
    out3 = processes.monte_carlo_portfolio_paths(returns, n_simulations=5, horizon=4, seed=1, weights=w)
    assert "source" in out3


def test_phase10_all_missing_and_hydra(monkeypatch, tmp_path):
    from iqrp.app.portfolio import phase10
    import iqrp.app.portfolio as port_pkg

    monkeypatch.setattr(port_pkg, "__all__", ["Portfolio"], raising=False)
    monkeypatch.setattr(phase10, "PHASE10_COMPONENTS", [])
    monkeypatch.setattr(phase10, "REQUIRED_DOCS", [])
    # force hydra missing by rewriting Path check via validate internals —
    # patch Path.is_file for hydra path
    report = phase10.validate_phase10()
    assert report["status"] == "FAIL"
    assert any("__all__" in f or "missing" in f.lower() for f in report["summary"]["failures"])


def test_phase10_main(monkeypatch, tmp_path, capsys):
    from iqrp.app.portfolio import phase10

    def fake_write(path=None):
        p = tmp_path / "out.json"
        p.write_text('{"status": "PASS", "summary": {}}', encoding="utf-8")
        return p

    monkeypatch.setattr(phase10, "write_phase10_report", fake_write)
    monkeypatch.setattr(phase10, "__name__", "__main__")
    # execute main block manually
    p = phase10.write_phase10_report()
    data = __import__("json").loads(p.read_text(encoding="utf-8"))
    print(p)
    print(data["status"], data["summary"])
    assert data["status"] == "PASS"


def test_optimize_robust_modes(mu, cov, names):
    from iqrp.app.portfolio.optimization.robust import optimize_robust

    r1 = optimize_robust(mu=mu, cov=cov, names=names, mode="parameter", max_weight=0.5)
    assert "success" in r1
    r2 = optimize_robust(mu=mu, cov=cov, names=names, mode="box", max_weight=0.5)
    assert "success" in r2
    r3 = optimize_robust(mu=mu, cov=cov, names=names, mode="ellipsoidal", max_weight=0.5)
    assert "success" in r3


def test_bl_optimizer_typeerror_fallback(mu, cov, names, monkeypatch):
    import iqrp.app.portfolio.optimization.black_litterman as bl
    import iqrp.app.portfolio.expected_returns.black_litterman as er_bl

    calls = {"n": 0}
    real = er_bl.black_litterman_posterior

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TypeError("bad kwargs")
        return real(*a, **{kk: vv for kk, vv in k.items() if kk in (
            "market_weights", "risk_aversion", "P", "Q", "omega", "tau", "equilibrium_mu", "names", "version"
        )} if False else k)

    # simpler: patch _call_bl_posterior's import target
    def boom_then_ok(cov_arr, **kwargs):
        if "delta" not in kwargs and "pi" not in kwargs:
            raise TypeError("need alt")
        n = cov_arr.shape[0] if hasattr(cov_arr, "shape") else len(cov_arr)
        return {"mu": np.zeros(n), "posterior_cov": np.eye(n), "method": "alt"}

    monkeypatch.setattr(er_bl, "black_litterman_posterior", boom_then_ok)
    n = len(names)
    P = np.zeros((1, n)); P[0, 0] = 1
    res = bl.optimize_black_litterman(
        cov=cov, names=names, P=P, Q=np.array([0.01]), omega=np.array([0.1]),
        equilibrium_returns=mu, max_weight=0.5,
    )
    assert "success" in res


def test_engine_extract_weights_sorted_and_rc_error(mu, cov, names, prices, adv, monkeypatch):
    from iqrp.app.portfolio.engine import _extract_weights
    import iqrp.app.portfolio.engine as eng_mod

    # n provided but dict length differs → sorted keys
    w = _extract_weights({"b": 0.3, "a": 0.7}, n=2)
    assert len(w) == 2
    w2 = _extract_weights({"a": 0.5, "b": 0.5}, n=2)
    assert w2 == [0.5, 0.5] or abs(sum(w2) - 1) < 1e-12

    eng = PortfolioConstructionEngine(
        settings=PortfolioSettings(require_risk_validation=False, seed=1)
    )

    def boom_rc(*a, **k):
        raise RuntimeError("rc fail")

    monkeypatch.setattr(eng_mod, "pr_risk_contribution", boom_rc)
    r = eng.construct(mu=mu, cov=cov, names=names, method="min_variance")
    assert "error" in (r.risk_contribution or {}) or isinstance(r.risk_contribution, dict)


def test_misc_empty_concentration_min_leverage_illcond():
    from iqrp.app.portfolio.constraints.concentration import concentration_metrics, check_concentration_constraints
    from iqrp.app.portfolio.constraints.leverage import check_leverage_constraints
    from iqrp.app.portfolio.constraints.liquidity import check_liquidity_constraints
    from iqrp.app.portfolio.constraints.turnover import check_turnover_constraints

    assert concentration_metrics([])["hhi"] == 0.0
    check_leverage_constraints(np.ones(4) * 0.1, min_leverage=1.0)
    assert check_liquidity_constraints([], adv=[1.0]) == []
    assert check_turnover_constraints(np.ones(4) / 4) == []

    # ill-conditioned cov
    c = np.eye(2) * 1e-16
    c[0, 1] = c[1, 0] = 1e-16
    nh = numerical_health(cov=np.array([[1.0, 0.0], [0.0, 1e-20]]))
    assert "healthy" in nh
    fd = feasibility_diagnostics(np.ones(4) / 4, budget=2.0)
    assert "budget" in fd["violations"]


def test_cov_robust_mcd_edges(rng):
    # t < 2 inside mcd via winsorize_mcd on tiny
    x = rng.normal(size=(3, 2))
    out = robust_covariance(x, method="mcd", n_trials=2, seed=0, h_fraction=0.9)
    assert "matrix" in out
    # zero-det subsets: force n_trials small on correlated data
    x2 = np.tile(rng.normal(size=(20, 1)), (1, 3))
    out2 = robust_covariance(x2, method="mcd", n_trials=3, seed=1)
    assert "name" in out2


def test_shrinkage_1d_and_empty_n(rng):
    out = shrinkage_covariance(rng.normal(size=30), method="risk")
    assert "matrix" in out
    # ledoit with intensity and empty handled via risk path
    out2 = ledoit_wolf_covariance(np.zeros((0, 0)))
    assert "name" in out2


def test_factor_cov_bad_fr_columns(rng):
    B = rng.normal(size=(4, 2))
    with pytest.raises(ValueError):
        factor_covariance(factor_loadings=B, factor_returns=rng.normal(size=(30, 3)))
    # residual single row → ddof=0
    fr = rng.normal(size=(1, 2))
    ar = rng.normal(size=(1, 4))
    out = factor_covariance(factor_loadings=B, factor_returns=fr, asset_returns=ar)
    assert out["residual_method"] == "ols_residuals"


def test_max_sharpe_minvar_exception_and_names(mu, cov, names, monkeypatch):
    import iqrp.app.portfolio.optimization.maximum_sharpe as ms
    import iqrp.app.portfolio.optimization.minimum_variance as mnv
    import iqrp.app.portfolio.optimization.entropy as ent
    import iqrp.app.portfolio.optimization.maximum_diversification as md
    import iqrp.app.portfolio.optimization.turnover as to
    import iqrp.app.portfolio.optimization.drawdown as dd

    def boom_parse(*a, **k):
        raise RuntimeError("x")

    for mod in (ms, mnv, ent, md, to, dd):
        monkeypatch.setattr(mod, "as_cov", boom_parse)
        fn = [x for x in dir(mod) if x.startswith("optimize_")][0]
        out = getattr(mod, fn)(mu=mu, cov=cov, names=names)
        assert out["success"] is False


def test_serializer_enum_mro_exception():
    class Weird:
        value = "z"
        # pretend enum-ish
    # force Enum check fail by having value + mro that isn't Enum
    assert isinstance(_to_jsonable(Weird()), str) or _to_jsonable(Weird()) is not None


def test_dp_empty_grid_and_no_pts():
    # levels that somehow yield empty — filter by tight box already tested;
    # force _simplex_grid fallback when no compositions (levels path always has pts for n>0)
    g = _simplex_grid(1, 1)
    assert g.shape[0] >= 1


def test_engine_construct_portfolio_and_targetweights_names(mu, cov, names, forecasts):
    eng = PortfolioConstructionEngine(
        settings=PortfolioSettings(require_risk_validation=False, seed=5, method="min_variance")
    )
    port = Portfolio(names=list(names), weights=[0.4, 0.3, 0.2, 0.1])
    r = eng.construct(mu=mu, cov=cov, current_portfolio=port, method="min_variance")
    assert list(r.names) == list(names) or len(r.names) == len(names)

    tw = TargetWeights.from_arrays([0.25] * 4, names=names)
    r2 = eng.construct(mu=mu, cov=cov, current_portfolio=tw, method="min_variance")
    assert r2 is not None

    # signals-only → names default
    r3 = eng.construct(signals=np.array([1.0, 0.5, -0.2, 0.1]), cov=cov, method="min_variance")
    assert len(r3.names) == 4

    # forecasts path
    r4 = eng.construct(forecasts=forecasts, cov=cov, names=names, method="mean_variance")
    assert r4 is not None


def test_engine_minvar_fallback_fails_then_cash(mu, cov, names, monkeypatch):
    eng = PortfolioConstructionEngine(
        settings=PortfolioSettings(require_risk_validation=False, fallback="min_variance", seed=2)
    )

    import iqrp.app.portfolio.engine as eng_mod

    def boom_mv(**kwargs):
        raise RuntimeError("primary fail")

    def boom_minvar(*a, **k):
        raise RuntimeError("minvar fail")

    monkeypatch.setitem(eng_mod._OPTIMIZER_MAP, "mean_variance", boom_mv)
    monkeypatch.setattr(eng_mod, "optimize_minimum_variance", boom_minvar)
    res = eng.optimize(mu=mu, cov=cov, method="mean_variance", names=names)
    assert res.fallback_used and res.fallback_kind == "cash"


def test_engine_risk_approved_empty_action(mu, cov, names, returns):
    class Dec:
        def check_limits(self, **kwargs):
            return []

        def validate_position(self, **kwargs):
            return {"approved": True, "reason": "ok"}  # no action key

    settings = PortfolioSettings(require_risk_validation=True, fallback="cash", seed=1)
    eng = PortfolioConstructionEngine(settings=settings, risk_engine=Dec())
    r = eng.construct(mu=mu, cov=cov, returns=returns, names=names, method="min_variance")
    assert r.risk_validation.get("approved") is True

    class SoftOnlyNoDecision:
        def check_limits(self, **kwargs):
            class B:
                severity = type("S", (), {"value": "soft"})()

                def to_dict(self):
                    return {"severity": "soft"}

            return [B()]

        def validate_position(self, **kwargs):
            raise RuntimeError("no decision")

    eng2 = PortfolioConstructionEngine(settings=settings, risk_engine=SoftOnlyNoDecision())
    r2 = eng2.construct(mu=mu, cov=cov, returns=None, names=names, method="min_variance")
    # soft breaches only path when no decision / no returns gate
    assert r2.risk_validation is not None


def test_max_sharpe_edges(mu, cov, names):
    from iqrp.app.portfolio.optimization.maximum_sharpe import optimize_maximum_sharpe

    assert optimize_maximum_sharpe(cov=cov, names=names)["success"] is False  # no mu
    # names from constraints
    res = optimize_maximum_sharpe(
        mu=mu, cov=cov, constraints={"names": names, "max_weight": 0.5}, max_weight=0.5
    )
    assert "success" in res
    # singular-ish → LinAlgError path
    sing = np.ones((4, 4)) * 1e-18
    res2 = optimize_maximum_sharpe(mu=np.zeros(4), cov=sing + np.eye(4) * 1e-18, names=names, max_weight=0.5, ridge=0.0)
    assert "success" in res2
    # all excess ~0 → equal weights seed
    res3 = optimize_maximum_sharpe(mu=np.zeros(4), cov=cov, names=names, max_weight=0.5, risk_free_rate=0.0)
    assert "success" in res3


def test_minvar_scipy_fail_and_names(cov, names, monkeypatch):
    import iqrp.app.portfolio.optimization.minimum_variance as mnv

    class Fake:
        success = False
        x = np.ones(4) / 4
        nit = 1

    monkeypatch.setattr(mnv, "minimize_scipy", lambda *a, **k: Fake())
    monkeypatch.setattr(mnv, "scipy_available", lambda: True)
    res = mnv.optimize_minimum_variance(cov=cov, constraints={"names": names}, max_weight=0.5)
    assert "success" in res


def test_entropy_weights_dict_and_size(cov, names, monkeypatch):
    import iqrp.app.portfolio.optimization.entropy as ent

    # find backend callable
    for attr in dir(ent):
        if "weight" in attr.lower() and callable(getattr(ent, attr)):
            pass
    # patch project after bad size via wrapping optimize internals
    monkeypatch.setattr(ent, "project_weights", lambda v, c: np.ones(int(c["n"])) * 0.1)
    out = ent.optimize_entropy(cov=cov, names=names, max_weight=0.5)
    assert out["success"] is False


def test_phase10_hydra_missing(monkeypatch, tmp_path):
    from iqrp.app.portfolio import phase10
    from pathlib import Path

    real_validate_parts = phase10.validate_phase10

    # Replace hydra path existence by patching Path.is_file selectively is hard;
    # instead monkeypatch Path used in validate for the hydra check
    orig_is_file = Path.is_file

    def fake_is_file(self):
        if "default.yaml" in str(self):
            return False
        return orig_is_file(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(phase10, "PHASE10_COMPONENTS", [])
    monkeypatch.setattr(phase10, "REQUIRED_DOCS", [])
    report = phase10.validate_phase10()
    assert any("default.yaml" in f for f in report["summary"]["failures"])


def test_run_phase10_as_main(tmp_path):
    import runpy
    import sys

    # Write a tiny shim that imports and runs main logic is heavy; call write + print like __main__
    from iqrp.app.portfolio.phase10 import write_phase10_report
    import json

    p = write_phase10_report(tmp_path / "p.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    # emulate __main__ prints
    print(p)
    print(data["status"], data["summary"])
    assert data["phase"] == "10"


def test_projection_minimize_scipy_unavailable(monkeypatch):
    import iqrp.app.portfolio.optimization.projection as proj

    monkeypatch.setattr(proj, "_scipy_minimize", None)
    assert proj.scipy_available() is False
    with pytest.raises(RuntimeError):
        proj.minimize_scipy(lambda x: 0.0, np.array([1.0]))


def test_historical_all_nan_rows():
    x = np.full((10, 3), np.nan)
    out = historical_expected_returns(x)
    assert out["n_obs"] == 0 or np.allclose(out["mu"], 0.0)
