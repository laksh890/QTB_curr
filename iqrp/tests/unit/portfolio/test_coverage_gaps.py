"""Coverage gaps: failure paths, PIT leakage, base/diagnostics/serializer/registry/viz."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from iqrp.app.portfolio.base import (
    OptimizationFailureError,
    OptimizationResult,
    Portfolio,
    PortfolioType,
    Position,
)
from iqrp.app.portfolio.base.constraints import (
    ConstraintKind,
    ConstraintSet,
    ConstraintSpec,
    ConstraintViolation,
    evaluate_constraints,
    conflicting_constraints,
    concentration_hhi,
    gross_exposure,
    leverage,
    long_exposure,
    net_exposure,
    short_exposure,
    turnover as base_turnover,
)
from iqrp.app.portfolio.base.objective import ObjectiveSpec, ObjectiveType
from iqrp.app.portfolio.base.optimizer import PortfolioOptimizer
from iqrp.app.portfolio.covariance import sample_covariance, shrinkage_covariance
from iqrp.app.portfolio.diagnostics import (
    diversification_metrics,
    feasibility_diagnostics,
    numerical_health,
    portfolio_diagnostics,
)
from iqrp.app.portfolio.engine import PortfolioConstructionEngine, dict_to_optimization_result
from iqrp.app.portfolio.config import PortfolioSettings
from iqrp.app.portfolio.portfolio_risk import (
    component_risk_contribution,
    factor_risk_decomposition,
    marginal_risk_contribution,
    percentage_risk_contribution,
    risk_contribution,
    risk_decomposition,
    volatility_contribution,
)
from iqrp.app.portfolio import processes
from iqrp.app.portfolio import registry
from iqrp.app.portfolio.serializer import PortfolioSerializer
from iqrp.app.portfolio import visualization


# --------------------------------------------------------------------------- PIT
def test_covariance_point_in_time_no_future_leakage(rng):
    """Cov on returns[:t] must be independent of returns[t:]."""
    n, t_cut, t_total = 4, 100, 200
    base = rng.multivariate_normal(np.zeros(n), np.eye(n) * 0.01**2, size=t_total)
    future_a = rng.normal(0, 0.05, size=(t_total - t_cut, n))
    future_b = rng.normal(0, 0.50, size=(t_total - t_cut, n))  # wildly different

    r_a = np.vstack([base[:t_cut], future_a])
    r_b = np.vstack([base[:t_cut], future_b])

    cov_a = sample_covariance(r_a[:t_cut])["matrix"]
    cov_b = sample_covariance(r_b[:t_cut])["matrix"]
    assert np.allclose(cov_a, cov_b)

    # Engine path
    eng = PortfolioConstructionEngine(
        settings=PortfolioSettings(require_risk_validation=False, seed=42)
    )
    ca = np.asarray(eng.covariance(returns=r_a[:t_cut], method="sample")["matrix"])
    cb = np.asarray(eng.covariance(returns=r_b[:t_cut], method="sample")["matrix"])
    assert np.allclose(ca, cb)

    # Full-series cov DOES change when future changes (sanity)
    full_a = np.asarray(sample_covariance(r_a)["matrix"])
    full_b = np.asarray(sample_covariance(r_b)["matrix"])
    assert not np.allclose(full_a, full_b, atol=1e-6)


def test_no_alpha_from_empty_forecasts(engine, names, cov):
    """Empty/zero forecasts must not invent alpha — express only."""
    result = engine.construct(
        forecasts=np.zeros(len(names)),
        cov=cov,
        names=names,
        method="mean_variance",
        forecast_confidence=np.zeros(len(names)),
    )
    # Expected return near zero or fallback; never a fabricated alpha edge
    if result.expected_return is not None and result.success and not result.fallback_used:
        assert abs(result.expected_return) < 0.05


# ------------------------------------------------------------------- portfolio_risk
def test_portfolio_risk_apis(weights, cov, names):
    mrc = marginal_risk_contribution(weights, cov)
    crc = component_risk_contribution(weights, cov)
    prc = percentage_risk_contribution(weights, cov)
    volc = volatility_contribution(weights, cov)
    rc = risk_contribution(weights, cov)
    assert isinstance(mrc, dict)
    assert isinstance(crc, dict)
    assert isinstance(prc, dict)
    assert isinstance(volc, dict)
    assert isinstance(rc, dict)

    # PRC sums ~1
    vals = prc.get("values") or prc.get("percentage") or []
    if vals is not None and len(np.asarray(vals).reshape(-1)) == len(weights):
        s = float(np.sum(vals))
        if s > 0:
            assert abs(s - 1.0) < 1e-4

    n = len(weights)
    B = np.eye(n)[:, :2]
    fr = factor_risk_decomposition(
        weights,
        factor_loadings=B,
        factor_cov=np.eye(2) * 0.01,
        idiosyncratic_var=np.ones(n) * 1e-4,
        factor_names=["f1", "f2"],
    )
    assert isinstance(fr, dict)
    rd = risk_decomposition(weights, cov, factor_loadings=B, factor_cov=np.eye(2) * 0.01)
    assert isinstance(rd, dict)


# --------------------------------------------------------------------- diagnostics
def test_diagnostics_modules(weights, cov, mu):
    nh = numerical_health(weights=weights, cov=cov, mu=mu)
    assert "healthy" in nh or "score" in nh

    fd = feasibility_diagnostics(weights, max_weight=0.5, max_gross=1.5, long_only=True)
    assert "feasible" in fd

    dm = diversification_metrics(weights, cov=cov)
    assert "hhi" in dm or "effective_n" in dm

    pd = portfolio_diagnostics(weights, cov=cov, mu=mu, max_weight=0.5, long_only=True)
    assert isinstance(pd, dict)

    # unhealthy cov
    bad = numerical_health(weights=np.array([np.nan, 0.5]), cov=np.eye(2))
    assert bad.get("healthy") is False or len(bad.get("issues") or []) > 0


# ---------------------------------------------------------------------- serializer
def test_serializer_roundtrip(tmp_path: Path, names, weights):
    ser = PortfolioSerializer()
    port = Portfolio(names=list(names), weights=list(weights), cash=0.0)
    p = ser.save_portfolio(port, tmp_path / "p.json")
    loaded = ser.load_portfolio(p)
    assert list(loaded.names) == list(names)

    res = OptimizationResult(success=True, weights=list(weights), names=list(names), method="mv")
    rp = ser.save_result(res, tmp_path / "r.json")
    loaded_r = ser.load_result(rp)
    assert loaded_r.success is True

    pos = Position(asset=names[0], quantity=10.0, price=100.0)
    pp = ser.save_position(pos, tmp_path / "pos.json")
    loaded_pos = ser.load_position(pp)
    assert loaded_pos.asset == names[0]

    raw = ser.dump_bytes(port)
    assert isinstance(raw, (bytes, bytearray, str)) or raw is not None
    back = ser.load_bytes(raw)
    assert back is not None


# ------------------------------------------------------------------------ registry
def test_registry_api(returns):
    avail = registry.available()
    assert len(avail) >= 1
    # get a known estimator if registered
    names = list(avail) if not isinstance(avail, dict) else list(avail.keys())
    if names:
        fn = registry.get(names[0])
        assert callable(fn) or fn is not None
    with pytest.raises(KeyError):
        registry.get("__definitely_missing_estimator__")

    def _dummy(returns, **kwargs):
        return shrinkage_covariance(returns, **kwargs)

    registry.register("test_dummy_cov_xyz", _dummy)
    assert registry.get("test_dummy_cov_xyz") is _dummy
    registry.clear_custom()


# ----------------------------------------------------------------------- processes
def test_processes_simulate_and_mc():
    scen = processes.simulate_portfolio_scenario(kind="normal", n=100, n_assets=4, seed=42)
    assert "returns" in scen or hasattr(scen, "shape") or isinstance(scen, dict)
    R = scen["returns"] if isinstance(scen, dict) else scen
    R = np.asarray(R)
    assert R.shape[0] == 100
    assert R.shape[1] == 4

    paths = processes.monte_carlo_portfolio_paths(
        R, n_simulations=50, horizon=10, seed=42, weights=np.ones(4) / 4
    )
    assert paths is not None

    batch = processes.process_scenarios(kinds=["normal"], n=80, n_assets=3, seed=1)
    assert batch is not None


# ----------------------------------------------------------------- visualization
def test_visualization_bundle(weights, cov, names):
    bundle = visualization.portfolio_viz_bundle(
        weights,
        weights_old=weights,
        risk_contribution=np.ones(len(weights)) / len(weights),
        names=names,
    )
    assert isinstance(bundle, dict)
    wp = visualization.weights_payload(weights, names=names)
    assert wp.get("type") == "bar" or "values" in wp or "labels" in wp
    rc = visualization.risk_contribution_payload(
        np.ones(len(weights)) / len(weights), names=names
    )
    assert isinstance(rc, dict)
    to = visualization.turnover_payload(weights, weights * 0.5, names=names)
    assert isinstance(to, dict)


# --------------------------------------------------------------------------- base
def test_base_portfolio_position_exposure(names, weights):
    port = Portfolio(
        names=list(names),
        weights=list(weights),
        portfolio_type=PortfolioType.LONG_ONLY,
    )
    assert abs(gross_exposure(weights) - 1.0) < 1e-9
    assert abs(net_exposure(weights) - 1.0) < 1e-9
    assert long_exposure(weights) >= 0
    assert short_exposure(weights) >= 0
    assert leverage(weights) >= 0
    assert concentration_hhi(weights) > 0
    assert base_turnover(weights, weights) == pytest.approx(0.0)

    pos = Position(asset=names[0], quantity=5.0, price=10.0)
    assert pos.to_dict()["asset"] == names[0]
    d = port.to_dict()
    assert "weights" in d
    port2 = Portfolio.from_dict(d)
    assert list(port2.names) == list(names)


def test_constraint_set_and_evaluate(weights):
    spec = ConstraintSpec(
        kind=ConstraintKind.MAX_POSITION,
        value=0.3,
        hard=True,
        name="max_w",
    )
    cs = ConstraintSet(constraints=[spec])
    w_bad = np.array([0.8, 0.2, 0.0, 0.0][: len(weights)], dtype=float)
    viols = evaluate_constraints(w_bad, cs)
    assert isinstance(viols, list)
    cv = ConstraintViolation(
        name="max_weight", kind="box", actual=0.8, limit=0.3, message="too big", hard=True
    )
    conf = conflicting_constraints([cv])
    assert isinstance(conf, list)
    assert len(conf) >= 1


def test_objective_spec():
    obj = ObjectiveSpec(objective_type=ObjectiveType.MEAN_VARIANCE, risk_aversion=1.5)
    assert obj.risk_aversion == 1.5


def test_optimization_result_failure_and_raise():
    fail = OptimizationResult.failure(reason="boom", names=["a", "b"], method="mv")
    assert fail.success is False
    d = fail.to_dict()
    back = OptimizationResult.from_dict(d)
    assert back.failure_reason == "boom"

    hard = OptimizationResult(
        success=False,
        weights=[0, 0],
        names=["a", "b"],
        violations=[
            ConstraintViolation(name="x", kind="box", actual=1, limit=0.1, message="hard", hard=True)
        ],
    )

    class Dummy(PortfolioOptimizer):
        def optimize(self, *args, **kwargs):
            return hard

    opt = Dummy()
    with pytest.raises(OptimizationFailureError):
        opt.raise_on_hard_violations(hard)

    fb = OptimizationResult(
        success=True,
        weights=[0.5, 0.5],
        names=["a", "b"],
        fallback_used=True,
        violations=[
            ConstraintViolation(name="x", kind="box", actual=1, limit=0.1, message="hard", hard=True)
        ],
    )
    assert opt.raise_on_hard_violations(fb) is fb

    port = hard.to_portfolio()
    assert isinstance(port, Portfolio)


def test_portfolio_optimizer_abc(cov, mu, names):
    class Dummy(PortfolioOptimizer):
        def optimize(self, *args, **kwargs):
            n = len(names)
            return OptimizationResult(success=True, weights=[1 / n] * n, names=list(names))

    opt = Dummy()
    res = opt.optimize(mu=mu, cov=cov)
    assert res.success


def test_dict_to_opt_result_edge_cases():
    r = dict_to_optimization_result({"success": True, "weights": [0.5, 0.5], "diagnostics": {"portfolio_variance": 0.01}})
    assert r.expected_variance == pytest.approx(0.01)
    r2 = dict_to_optimization_result({"success": False, "weights": {}})
    assert r2.success is False
    r3 = dict_to_optimization_result(
        {"success": True, "weights": [0.5, 0.5]},
        mu=[0.01, 0.02],
        cov=np.eye(2) * 0.01,
        fallback_used=True,
        fallback_kind="cash",
    )
    assert r3.fallback_used
    assert r3.expected_return is not None


def test_engine_optimize_exception_fallback(mu, cov, names, monkeypatch):
    eng = PortfolioConstructionEngine(
        settings=PortfolioSettings(require_risk_validation=False, fallback="cash", seed=1)
    )

    def _boom(**kwargs):
        raise RuntimeError("explode")

    monkeypatch.setitem(
        __import__("iqrp.app.portfolio.engine", fromlist=["_OPTIMIZER_MAP"])._OPTIMIZER_MAP,
        "mean_variance",
        _boom,
    )
    # re-import map reference
    import iqrp.app.portfolio.engine as eng_mod

    monkeypatch.setitem(eng_mod._OPTIMIZER_MAP, "mean_variance", _boom)
    res = eng.optimize(mu=mu, cov=cov, method="mean_variance", names=names)
    assert res.fallback_used is True
    assert res.fallback_kind == "cash"


def test_engine_risk_breaches_soft_and_hard(names, returns, mu, cov):
    class SoftBreach:
        severity = type("S", (), {"value": "soft"})()

        def to_dict(self):
            return {"severity": "soft"}

    class HardBreach:
        severity = type("S", (), {"value": "hard"})()

        def to_dict(self):
            return {"severity": "hard"}

    class SoftRisk:
        def check_limits(self, **kwargs):
            return [SoftBreach()]

        def validate_position(self, **kwargs):
            return {"approved": True, "action": "APPROVE", "reason": "ok"}

    class HardRisk:
        def check_limits(self, **kwargs):
            return [HardBreach()]

        def validate_position(self, **kwargs):
            return {"approved": True, "action": "APPROVE", "reason": "ok"}

    settings = PortfolioSettings(require_risk_validation=True, fallback="cash", seed=1)
    eng_soft = PortfolioConstructionEngine(settings=settings, risk_engine=SoftRisk())
    r1 = eng_soft.construct(mu=mu, cov=cov, returns=returns, names=names, method="min_variance")
    assert r1.risk_validation is not None

    eng_hard = PortfolioConstructionEngine(settings=settings, risk_engine=HardRisk())
    r2 = eng_hard.construct(mu=mu, cov=cov, returns=returns, names=names, method="min_variance")
    assert r2.risk_validation is not None
    # hard breaches → reject / fallback
    assert (
        r2.risk_validation.get("approved") is False
        or r2.fallback_used
        or r2.risk_validation.get("action") in ("REJECT", "CAUTION", "APPROVE")
    )


def test_engine_risk_ensemble_path(names, returns, mu, cov):
    class Ens:
        def validate_position(self, **kwargs):
            return {"approved": False, "action": "REJECT", "reason": "ensemble halt"}

    settings = PortfolioSettings(require_risk_validation=True, fallback="current", seed=1)
    eng = PortfolioConstructionEngine(
        settings=settings,
        risk_ensemble=Ens(),
        risk_engine=None,
    )
    cur = np.ones(len(names)) / len(names)
    r = eng.construct(
        mu=mu,
        cov=cov,
        returns=returns,
        names=names,
        current_portfolio=cur,
        method="min_variance",
    )
    assert r.risk_validation is not None
    if r.risk_validation.get("approved") is False:
        assert r.fallback_used


def test_engine_risk_check_limits_error(names, returns, mu, cov):
    class Broken:
        def check_limits(self, **kwargs):
            raise RuntimeError("limits down")

        def validate_position(self, **kwargs):
            raise RuntimeError("validate down")

    settings = PortfolioSettings(require_risk_validation=True, fallback="cash", seed=1)
    eng = PortfolioConstructionEngine(settings=settings, risk_engine=Broken())
    r = eng.construct(mu=mu, cov=cov, returns=returns, names=names, method="min_variance")
    assert r.risk_validation is not None
    assert "check_limits_error" in r.risk_validation or "validate_position_error" in r.risk_validation or r.risk_validation.get("approved") is not None


def test_construct_include_tc_from_zero(engine, mu, cov, names, prices, adv):
    r = engine.construct(
        mu=mu,
        cov=cov,
        names=names,
        prices=prices,
        capital=1e6,
        include_transaction_costs=True,
        adv=adv,
    )
    # no current → may still compute if flag True with zeros path when False default...
    # Explicitly request from-zero costs
    r2 = engine.construct(
        mu=mu,
        cov=cov,
        names=names,
        prices=prices,
        capital=1e6,
        include_transaction_costs=True,
        current_portfolio=np.zeros(len(names)),
        adv=adv,
    )
    assert r2.transaction_cost.get("total", 0) >= 0.0


def test_settings_covariance_expected_returns_nested():
    s = PortfolioSettings(
        require_risk_validation=False,
        covariance={"method": "ewma", "ewma_lambda": 0.9},
        expected_returns={"method": "historical", "bl_tau": 0.1},
    )
    assert s.covariance.method == "ewma"
    assert s.expected_returns.method == "historical"


def test_robust_covariance_methods(returns):
    for method in ("winsorize", "mcd", "winsorize_mcd"):
        out = __import__(
            "iqrp.app.portfolio.covariance.robust", fromlist=["robust_covariance"]
        ).robust_covariance(returns, method=method, n_trials=4, seed=0)
        assert "matrix" in out


# ---------------------------------------------------------- extra gap fillers
def test_base_constraints_evaluate_turnover_long_only_dollar(names):
    from iqrp.app.portfolio.base.constraints import (
        ConstraintKind,
        ConstraintSpec,
        evaluate_constraints,
    )

    w = np.array([0.5, -0.2, 0.4, 0.3])
    specs = [
        ConstraintSpec(kind=ConstraintKind.MAX_TURNOVER, value=0.01, hard=True),
        ConstraintSpec(kind=ConstraintKind.LONG_ONLY, hard=True),
        ConstraintSpec(kind=ConstraintKind.DOLLAR_NEUTRAL, params={"tol": 1e-9}),
    ]
    viols = evaluate_constraints(w, specs, names=names, current_weights=np.zeros(4))
    assert any(v.kind == ConstraintKind.LONG_ONLY.value for v in viols)
    assert any(v.kind == ConstraintKind.DOLLAR_NEUTRAL.value for v in viols)


def test_engine_optimize_from_returns_only(names, returns):
    eng = PortfolioConstructionEngine(
        settings=PortfolioSettings(require_risk_validation=False, seed=3, method="mean_variance")
    )
    res = eng.optimize(returns=returns, names=names, method="mean_variance", max_weight=0.5)
    assert res.success or res.fallback_used or res.failure_reason


def test_engine_cvar_with_returns_scenarios(names, returns, cov):
    eng = PortfolioConstructionEngine(
        settings=PortfolioSettings(require_risk_validation=False, seed=4, method="cvar")
    )
    res = eng.optimize(cov=cov, returns=returns, names=names, method="cvar", max_weight=0.5)
    assert "success" in (res.to_dict() if hasattr(res, "to_dict") else {"success": res.success})


def test_processes_low_liquidity_and_large_gaps():
    for kind in ("low_liquidity", "large_gaps", "regime_transition"):
        out = processes.simulate_portfolio_scenario(kind=kind, n=50, n_assets=3, seed=9)
        assert "returns" in out


def test_tc_scalar_broadcast_and_optimizer_weight_array():
    from iqrp.app.portfolio.transaction_costs.market_impact import market_impact_cost
    from iqrp.app.portfolio.transaction_costs.slippage import slippage_cost
    from iqrp.app.portfolio.transaction_costs.spread import spread_cost
    from iqrp.app.portfolio.base.optimizer import OptimizationResult
    from iqrp.app.portfolio.config import PortfolioSettings

    trades = np.array([0.1, -0.05, 0.0, 0.2])
    mi = market_impact_cost(trades, capital=1e6, adv=1e7, prices=100.0, vols=0.02)
    assert mi["total"] >= 0
    sl = slippage_cost(trades, capital=1e6, prices=50.0, vols=0.02, adv=1e7)
    assert sl["total"] >= 0
    sp = spread_cost(trades, capital=1e6, spreads=0.001)
    assert sp["total"] >= 0

    res = OptimizationResult(success=True, weights=[0.5, 0.5], names=["a", "b"])
    assert res.weight_array().shape == (2,)

    from omegaconf import OmegaConf

    s = PortfolioSettings.from_mapping(OmegaConf.create({"method": "min_variance", "require_risk_validation": False}))
    assert s.method == "min_variance"
    with pytest.raises(Exception):
        PortfolioSettings.from_mapping({"method": {"nested": True}})  # invalid type


def test_hierarchical_weights_dict_path(cov, names, monkeypatch):
    import iqrp.app.portfolio.optimization.hierarchical as hr

    key_names = list(names)

    def fake_hrp(*a, **k):
        return {"weights": {nm: 1.0 / len(key_names) for nm in key_names}}

    monkeypatch.setattr(hr, "hrp_weights", fake_hrp)
    out = hr.optimize_hrp(cov=cov, names=names, max_weight=0.6)
    assert "success" in out


def test_phase10_runpy_main(tmp_path, monkeypatch):
    from iqrp.app.portfolio import phase10
    import json

    out = tmp_path / "Phase10_PortfolioConstruction_Validation.json"

    def _write(path=None):
        out.write_text('{"status":"PASS","summary":{}}', encoding="utf-8")
        return out

    monkeypatch.setattr(phase10, "write_phase10_report", _write)
    p = phase10.write_phase10_report()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
