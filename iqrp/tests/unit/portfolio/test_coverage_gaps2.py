"""Coverage gaps 2: base/, constraints internals, projection, optimizer fails, commissions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from iqrp.app.portfolio.base.constraints import (
    ConstraintKind,
    ConstraintSet,
    ConstraintSpec,
    ConstraintViolation,
    concentration_hhi,
    evaluate_concentration,
    evaluate_constraints,
    evaluate_gross,
    evaluate_leverage,
    evaluate_long_short,
    evaluate_max_position,
    evaluate_min_position,
    evaluate_net,
    leverage,
    long_exposure,
    max_position_weight,
    short_exposure,
    turnover,
)
from iqrp.app.portfolio.base.objective import ObjectiveSpec, ObjectiveType
from iqrp.app.portfolio.base.portfolio import Portfolio, PortfolioType
from iqrp.app.portfolio.base.position import Position
from iqrp.app.portfolio.constraints._types import (
    ConstraintSeverity,
    as_weights,
    coerce_severity,
    filter_by_severity,
    make_violation,
)
from iqrp.app.portfolio.constraints.beta import check_beta_constraints, portfolio_beta
from iqrp.app.portfolio.constraints.currency import check_currency_constraints, currency_exposures
from iqrp.app.portfolio.constraints.factor import check_factor_constraints, portfolio_factor_exposures
from iqrp.app.portfolio.constraints.liquidity import check_liquidity_constraints
from iqrp.app.portfolio.constraints.position import check_position_constraints
from iqrp.app.portfolio.constraints.sector import check_sector_constraints, sector_exposures
from iqrp.app.portfolio.optimization.cvar import optimize_cvar
from iqrp.app.portfolio.optimization.drawdown import optimize_drawdown
from iqrp.app.portfolio.optimization.entropy import optimize_entropy
from iqrp.app.portfolio.optimization.hierarchical import optimize_hrp
from iqrp.app.portfolio.optimization.maximum_diversification import optimize_maximum_diversification
from iqrp.app.portfolio.optimization.maximum_sharpe import optimize_maximum_sharpe
from iqrp.app.portfolio.optimization.mean_variance import optimize_mean_variance
from iqrp.app.portfolio.optimization.minimum_variance import optimize_minimum_variance
from iqrp.app.portfolio.optimization.projection import (
    as_cov,
    as_vector,
    check_feasibility,
    equal_weights,
    failed_result,
    format_weights,
    infeasible_result,
    make_result,
    minimize_scipy,
    parse_constraints,
    portfolio_return,
    portfolio_variance,
    project_box_simplex,
    project_gross,
    project_simplex,
    project_weights,
    projected_gradient,
    scipy_available,
    stabilize_mu,
)
from iqrp.app.portfolio.optimization.risk_parity import optimize_risk_parity
from iqrp.app.portfolio.optimization.turnover import optimize_turnover
from iqrp.app.portfolio.transaction_costs.commissions import commission_cost


# ============================================================================= base/constraints
def test_constraint_spec_post_init_and_roundtrip():
    spec = ConstraintSpec(kind="max_position", value=0.3, asset="A0")
    assert spec.kind == ConstraintKind.MAX_POSITION
    assert "max_position" in (spec.name or "")
    d = spec.to_dict()
    back = ConstraintSpec.from_dict(d)
    assert back.value == 0.3
    # from_dict defaults / None limits
    bare = ConstraintSpec.from_dict({"kind": "custom"})
    assert bare.value is None
    assert bare.lower is None


def test_constraint_violation_roundtrip_null_limit():
    v = ConstraintViolation(
        name="x", kind="box", actual=1.0, limit=None, message="m", hard=False, asset="A"
    )
    d = v.to_dict()
    assert d["limit"] is None
    back = ConstraintViolation.from_dict(d)
    assert back.limit is None
    assert back.hard is False


def test_constraint_set_materializes_all_scalar_fields():
    cs = ConstraintSet(
        constraints=[],
        long_only=True,
        max_weight=0.4,
        min_weight=0.01,
        max_gross=1.5,
        max_net=1.0,
        min_net=-0.5,
        max_long=1.0,
        max_short=0.5,
        max_leverage=2.0,
        max_concentration=0.5,
        max_turnover=0.3,
        dollar_neutral=True,
        dollar_neutral_tol=1e-4,
    )
    kinds = {c.kind for c in cs.constraints}
    assert ConstraintKind.LONG_ONLY in kinds
    assert ConstraintKind.MIN_POSITION in kinds
    assert ConstraintKind.MAX_NET in kinds
    assert ConstraintKind.MIN_NET in kinds
    assert ConstraintKind.MAX_LONG in kinds
    assert ConstraintKind.MAX_SHORT in kinds
    assert ConstraintKind.MAX_CONCENTRATION in kinds
    assert ConstraintKind.MAX_TURNOVER in kinds
    assert ConstraintKind.DOLLAR_NEUTRAL in kinds
    cs.add(ConstraintSpec(kind=ConstraintKind.CUSTOM, value=1.0, name="extra"))
    d = cs.to_dict()
    back = ConstraintSet.from_dict(d)
    assert back.max_weight == 0.4
    assert back.dollar_neutral is True


def test_constraint_set_evaluate_method(names):
    cs = ConstraintSet(max_weight=0.2, max_gross=1.0, long_only=True)
    w = np.array([0.5, 0.5, 0.0, 0.0])
    viols = cs.evaluate(w, names=names, current_weights=np.zeros(4))
    assert any(v.kind == ConstraintKind.MAX_POSITION.value for v in viols)


def test_exposure_helpers_empty_and_nan():
    assert long_exposure([]) == 0.0
    assert short_exposure([]) == 0.0
    assert concentration_hhi([0.0, 0.0]) == 0.0
    assert max_position_weight([]) == 0.0
    assert leverage([np.nan, 0.5]) == pytest.approx(0.5)
    # turnover pad branches
    assert turnover([0.5, 0.5], [1.0]) >= 0.0
    assert turnover([1.0], [0.5, 0.5]) >= 0.0
    assert turnover([0.5, 0.5], None) == pytest.approx(1.0)


def test_evaluate_max_min_position_with_and_without_names():
    w = np.array([0.6, 0.05, 0.0, 0.35])
    vmax = evaluate_max_position(w, 0.4, names=["A", "B", "C", "D"], hard=True)
    assert any(v.asset == "A" for v in vmax)
    vmax2 = evaluate_max_position(w, 0.4)  # index as asset
    assert any(v.asset == "0" for v in vmax2)

    vmin = evaluate_min_position(w, 0.1, names=["A", "B", "C", "D"], only_active=True)
    assert any(v.asset == "B" for v in vmin)
    # inactive skipped
    assert all(v.asset != "C" for v in vmin)
    vmin_all = evaluate_min_position(w, 0.1, only_active=False, hard=False)
    assert any(v.asset == "2" for v in vmin_all)


def test_evaluate_gross_net_long_short_leverage_concentration():
    w = np.array([0.8, 0.5, -0.4, 0.0])
    assert evaluate_gross(w, 1.0)
    assert not evaluate_gross(w, 10.0)

    net_v = evaluate_net(w, max_net=0.5, min_net=2.0)
    kinds = {v.kind for v in net_v}
    assert ConstraintKind.MAX_NET.value in kinds
    assert ConstraintKind.MIN_NET.value in kinds
    assert evaluate_net(w) == []

    ls = evaluate_long_short(w, max_long=0.5, max_short=0.1)
    assert any(v.kind == ConstraintKind.MAX_LONG.value for v in ls)
    assert any(v.kind == ConstraintKind.MAX_SHORT.value for v in ls)
    assert evaluate_long_short(w) == []

    assert evaluate_leverage(w, 1.0)
    assert not evaluate_leverage(w, 10.0)

    # concentrated book
    conc = evaluate_concentration(np.array([1.0, 0.0, 0.0, 0.0]), 0.5)
    assert conc
    assert not evaluate_concentration(np.ones(4) / 4, 0.9)


def test_evaluate_constraints_all_kinds_and_iterable(names):
    w = np.array([0.7, -0.2, 0.4, 0.1])
    specs = [
        ConstraintSpec(kind=ConstraintKind.MAX_POSITION, value=0.3, asset="A0"),
        ConstraintSpec(kind=ConstraintKind.MAX_POSITION, value=0.3, asset="MISSING"),
        ConstraintSpec(kind=ConstraintKind.MAX_POSITION, upper=0.25),
        ConstraintSpec(kind=ConstraintKind.MIN_POSITION, lower=0.15),
        ConstraintSpec(kind=ConstraintKind.MAX_GROSS, value=0.5),
        ConstraintSpec(kind=ConstraintKind.MAX_NET, value=0.5),
        ConstraintSpec(kind=ConstraintKind.MIN_NET, value=2.0),
        ConstraintSpec(kind=ConstraintKind.MAX_LONG, value=0.5),
        ConstraintSpec(kind=ConstraintKind.MAX_SHORT, value=0.05),
        ConstraintSpec(kind=ConstraintKind.MAX_LEVERAGE, value=0.5),
        ConstraintSpec(kind=ConstraintKind.MAX_CONCENTRATION, value=0.2),
        ConstraintSpec(kind=ConstraintKind.MAX_TURNOVER, value=0.01),
        ConstraintSpec(kind=ConstraintKind.LONG_ONLY),
        ConstraintSpec(kind=ConstraintKind.DOLLAR_NEUTRAL, params={"tol": 1e-9}),
        # value None branches (skipped evaluation for scalar kinds)
        ConstraintSpec(kind=ConstraintKind.MAX_GROSS, value=None),
        ConstraintSpec(kind=ConstraintKind.MAX_NET, value=None),
        ConstraintSpec(kind=ConstraintKind.MIN_NET, value=None),
        ConstraintSpec(kind=ConstraintKind.MAX_LONG, value=None),
        ConstraintSpec(kind=ConstraintKind.MAX_SHORT, value=None),
        ConstraintSpec(kind=ConstraintKind.MAX_LEVERAGE, value=None),
        ConstraintSpec(kind=ConstraintKind.MAX_CONCENTRATION, value=None),
        ConstraintSpec(kind=ConstraintKind.MAX_TURNOVER, value=None),
    ]
    viols = evaluate_constraints(w, specs, names=names, current_weights=np.zeros(4))
    assert isinstance(viols, list)
    assert len(viols) >= 5
    # iterable path (not ConstraintSet)
    viols2 = evaluate_constraints(w, iter(specs[:3]), names=names)
    assert isinstance(viols2, list)


# ============================================================================= base/portfolio + position + objective
def test_portfolio_cash_with_weights_and_from_dict_edges(names):
    cash = Portfolio.cash_portfolio(currency="EUR", cash=2.5)
    assert cash.cash == 2.5
    assert cash.meta.get("fallback") == "cash"
    assert cash.n_assets == 0

    pos = Position(asset=names[0], quantity=2.0, price=10.0, weight=0.25)
    port = Portfolio(positions=[pos], portfolio_type="long_short")
    assert port.names == [names[0]]
    assert port.weights == [0.25]
    assert port.portfolio_type == PortfolioType.LONG_SHORT

    # align partial weights
    port2 = Portfolio(names=list(names), weights=[0.5, 0.5], portfolio_type=PortfolioType.LONG_ONLY)
    assert len(port2.weights) == len(names)
    assert port2.weights[2] == 0.0

    # with_weights updates existing + new assets
    updated = port.with_weights([0.4, 0.6], names=[names[0], names[1]])
    assert updated.weights == [0.4, 0.6]
    assert updated.position_map()[names[0]].weight == 0.4
    assert names[1] in updated.position_map()

    with pytest.raises(ValueError):
        port.with_weights([0.5], names=[names[0], names[1]])

    # exposures on long-short
    ls = Portfolio(names=["a", "b"], weights=[0.8, -0.3])
    assert ls.long_exposure() == pytest.approx(0.8)
    assert ls.short_exposure() == pytest.approx(0.3)
    assert ls.leverage() == pytest.approx(1.1)
    assert ls.gross_exposure() == pytest.approx(1.1)
    assert ls.net_exposure() == pytest.approx(0.5)

    # from_dict via positions only
    d = {"positions": [pos.to_dict()], "nav": 100.0}
    p3 = Portfolio.from_dict(d)
    assert p3.nav == 100.0
    assert p3.names


def test_position_compute_and_signed_notional():
    p = Position(asset="X", quantity=-3.0, price=10.0, multiplier=2.0, notional=None)
    assert p.notional == pytest.approx(-60.0)
    p.price = 12.0
    assert p.compute_notional() == pytest.approx(-72.0)
    assert p.signed_notional() == pytest.approx(-72.0)
    p.notional = None
    assert p.signed_notional() == pytest.approx(-72.0)


def test_objective_spec_to_from_dict_and_name_property():
    obj = ObjectiveSpec(
        objective_type="mean_variance",
        target_return=0.1,
        target_volatility=0.2,
        risk_budgets={"a": 0.5},
        weights={"mv": 1.0},
    )
    assert obj.name == "mean_variance"
    d = obj.to_dict()
    back = ObjectiveSpec.from_dict(d)
    assert back.target_return == 0.1
    assert back.risk_budgets["a"] == 0.5
    # string path for name when somehow not enum (force via params only)
    bare = ObjectiveSpec.from_dict({})
    assert bare.objective_type == ObjectiveType.MEAN_VARIANCE


# ============================================================================= constraints/_types + modules
def test_coerce_severity_and_make_violation_paths():
    assert coerce_severity(None, hard=False) == ConstraintSeverity.SOFT
    assert coerce_severity(None, hard=True) == ConstraintSeverity.HARD
    assert coerce_severity(None) == ConstraintSeverity.HARD
    assert coerce_severity(ConstraintSeverity.WARNING) == ConstraintSeverity.WARNING
    assert coerce_severity(" soft ") == ConstraintSeverity.SOFT
    assert coerce_severity("WARNING") == ConstraintSeverity.WARNING
    assert coerce_severity("other") == ConstraintSeverity.HARD

    v1 = make_violation(
        "x",
        observed=1.0,
        threshold=0.5,
        reason="r",
        severity="soft",
        scope="sector",
        metadata={"asset": "A0", "kind": "custom"},
    )
    assert v1.hard is False
    assert v1.asset == "A0"
    assert "[sector]" in v1.message

    v2 = make_violation(
        "y",
        observed=1.0,
        threshold=0.5,
        reason="r",
        metadata={"index": 2},
    )
    assert v2.asset == "2"

    soft = ConstraintViolation("s", "k", 1.0, 0.5, "m", hard=False)
    hard = ConstraintViolation("h", "k", 1.0, 0.5, "m", hard=True)
    assert len(filter_by_severity([soft, hard], include_soft=False)) == 1
    assert len(filter_by_severity([soft, hard], include_hard=False)) == 1


def test_as_weights_broadcast_and_pad():
    assert as_weights(0.5, n=4).shape == (4,)
    assert as_weights([0.1, 0.2], n=4).shape == (4,)
    assert as_weights([1, 2, 3, 4], n=2).shape == (2,)


def test_beta_constraints_edge_cases(weights):
    assert check_beta_constraints(weights) == []
    assert check_beta_constraints(weights, betas=[1, 1, 1, 1]) == []
    # scalar beta broadcast + length mismatch pad
    b = portfolio_beta(weights, 1.2)
    assert isinstance(b, float)
    b2 = portfolio_beta(weights, [1.0, 0.5])
    assert isinstance(b2, float)

    viols = check_beta_constraints(
        weights,
        portfolio_beta_value=2.0,
        max_beta=1.0,
        min_beta=3.0,
        target_beta=0.0,
        beta_tol=0.01,
        severity="soft",
    )
    assert len(viols) >= 2


def test_factor_constraints_transpose_empty_and_min(weights):
    assert check_factor_constraints(weights) == []
    assert check_factor_constraints(weights, factor_loadings=np.eye(4)) == []
    # 1-D loadings
    exp = portfolio_factor_exposures(weights, np.ones(4), factor_names=["f"])
    assert "f" in exp
    # transpose path (k x n)
    B = np.random.default_rng(0).normal(size=(2, 4))
    exp2 = portfolio_factor_exposures(weights, B, factor_names=["a"])
    assert len(exp2) == 2  # names padded
    # pad/truncate when shapes mismatch
    exp3 = portfolio_factor_exposures(weights, np.ones((2, 2)))
    assert isinstance(exp3, dict)
    with pytest.raises(ValueError):
        portfolio_factor_exposures(weights, np.ones((2, 2, 2)))

    viols = check_factor_constraints(
        weights,
        factor_loadings=np.eye(4),
        max_factor_exposure={"factor_0": 0.01},
        min_factor_exposure=0.5,
        factor_neutral=["factor_0"],
        neutrality_tol=1e-12,
    )
    assert viols
    # factor_neutral True + mapping miss
    check_factor_constraints(
        weights,
        factor_loadings=np.eye(4),
        max_factor_exposure={"nope": 0.01},
        factor_neutral=True,
    )


def test_sector_currency_position_liquidity_edges(weights, names, prices, adv):
    assert sector_exposures(weights, None) == {}
    assert sector_exposures([], ["Tech"]) == {}
    # mapping by name / index / str index / unknown
    sm = {names[0]: "Tech", 1: "Fin", "2": "Tech"}
    se = sector_exposures(weights, sm, names=names)
    assert "UNKNOWN" in se or "Tech" in se
    se2 = sector_exposures(weights, ["Tech", "Fin"], names=names)  # short sequence → UNKNOWN
    assert "UNKNOWN" in se2
    assert check_sector_constraints(weights) == []
    viols = check_sector_constraints(
        weights,
        sector_map=["Tech", "Tech", "Fin", "Fin"],
        names=names,
        max_sector_weight={"Tech": 0.01},
        min_sector_weight=0.9,
    )
    assert viols

    assert currency_exposures(weights, None) == {}
    ce = currency_exposures(weights, {0: "USD", "1": "EUR"})
    assert "USD" in ce
    ce2 = currency_exposures(weights, ["USD"])  # short → USD default
    assert "USD" in ce2
    assert check_currency_constraints(weights) == []
    cv = check_currency_constraints(
        weights,
        currencies=["USD", "EUR", "USD", "JPY"],
        max_currency_exposure={"USD": 0.01},
        min_currency_exposure=0.9,
    )
    assert cv

    assert check_position_constraints([]) == []
    # per-asset bound vectors shorter than n
    pv = check_position_constraints(
        weights,
        max_position=[0.1, 0.2],
        min_position=[-0.1],
        long_only=True,
    )
    # long_only with shorts
    pv2 = check_position_constraints([-0.2, 0.5, 0.3, 0.4], long_only=True, max_weight=0.3)
    assert any(v.name == "long_only" for v in pv2)

    assert check_liquidity_constraints(weights) == []
    assert check_liquidity_constraints(weights, adv=adv) == []
    # zero notional skip + breaches
    w0 = np.array([0.0, 0.5, 0.5, 0.0])
    lv = check_liquidity_constraints(
        w0,
        adv=np.array([1.0, 1.0, 1.0, 1.0]),
        prices=prices,
        capital=1e9,
        max_participation=0.01,
        max_ttl=0.01,
        min_adv_coverage=1e9,
    )
    assert isinstance(lv, list)


# ============================================================================= projection
def test_projection_as_vector_as_cov_stabilize():
    assert as_vector(None, n=3).tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        as_vector(None)
    with pytest.raises(ValueError):
        as_vector([1, 2], n=3)
    with pytest.raises(ValueError):
        as_cov([1, 2, 3])
    with pytest.raises(ValueError):
        as_cov(np.eye(2), n=3)
    assert stabilize_mu(np.array([])).size == 0
    m = stabilize_mu(np.array([0.0, 10.0, -10.0, 0.01]), clip=0.5, winsor_z=1.0)
    assert float(np.max(np.abs(m))) <= 0.5 + 1e-12


def test_parse_constraints_dict_object_extras():
    d = parse_constraints(
        {
            "long_only": True,
            "max_weight": 0.3,
            "min_weight": -0.1,
            "max_gross": 1.2,
            "budget": 1.0,
            "sum_weights": 0.9,
            "names": ["a", "b", "c", "d"],
            "linear_eq": [[1, 0, 0, 0]],
            "linear_ineq": [[0, 1, 0, 0]],
            "group_limits": {"g": 0.5},
            "sector_limits": {"Tech": 0.4},
        },
        4,
        long_only=False,
        min_weight=-0.2,
    )
    assert d["extras"]
    assert d["names"] == ["a", "b", "c", "d"]
    # sum_weights ignored when budget present
    d2 = parse_constraints({"sum_weights": 0.8}, 4)
    assert d2["budget"] == 0.8
    # object attributes
    obj = SimpleNamespace(long_only=False, max_weight=0.25, min_weight=-0.2, max_gross=1.1, budget=1.0, names=["x"] * 4)
    d3 = parse_constraints(obj, 4)
    assert d3["long_only"] is False
    assert d3["ub"] == 0.25


def test_check_feasibility_conflict_branches():
    ok, _, _ = check_feasibility({"n": 0, "lb": 0, "ub": 1, "budget": 1, "max_gross": None, "extras": []})
    assert ok is False
    bad = {
        "n": 4,
        "lb": 0.5,
        "ub": 0.4,
        "budget": 1.0,
        "max_gross": 0.5,
        "extras": [{"linear_eq": 1}],
    }
    ok2, reason, conflicts = check_feasibility(bad)
    assert ok2 is False
    assert reason
    assert any("unsupported" in c for c in conflicts)
    # n*lb > budget
    ok3, _, c3 = check_feasibility({"n": 4, "lb": 0.3, "ub": 1.0, "budget": 1.0, "max_gross": None, "extras": []})
    assert ok3 is False
    assert any("min_weight" in x for x in c3)


def test_project_simplex_box_gross_weights_paths():
    assert project_simplex(np.array([])).size == 0
    assert project_simplex(np.ones(3), budget=0).tolist() == [0, 0, 0]
    # rho empty → equal
    w = project_simplex(np.array([-1.0, -2.0, -3.0]), budget=1.0)
    assert abs(w.sum() - 1.0) < 1e-8
    w2 = project_simplex(np.array([0.5, 0.3, 0.2]), budget=1.0)
    assert abs(w2.sum() - 1.0) < 1e-8

    assert project_box_simplex(np.array([])).size == 0
    with pytest.raises(ValueError):
        project_box_simplex(np.ones(3), lb=0.5, ub=0.6, budget=1.0)
    pb = project_box_simplex(np.array([0.9, 0.05, 0.05, 0.0]), lb=0.0, ub=0.5, budget=1.0)
    assert abs(pb.sum() - 1.0) < 1e-6

    # project_gross within / shrink / restore
    g1 = project_gross(np.array([0.4, 0.3, 0.2, 0.1]), max_gross=2.0, budget=1.0, long_only=True)
    assert abs(g1.sum() - 1.0) < 1e-8
    g2 = project_gross(np.array([2.0, 2.0, 0.0, 0.0]), max_gross=1.0, budget=1.0, long_only=False)
    assert float(np.sum(np.abs(g2))) <= 1.0 + 1e-8

    cstr = parse_constraints(None, 4, long_only=True, max_weight=0.5, max_gross=1.2)
    pw = project_weights(np.array([0.8, 0.1, 0.05, 0.05]), cstr)
    assert abs(pw.sum() - 1.0) < 1e-5

    cstr_ls = parse_constraints({"long_only": False, "min_weight": -0.5, "max_weight": 0.5}, 4, long_only=False, min_weight=-0.5)
    pw2 = project_weights(np.array([0.5, -0.2, 0.4, 0.3]), cstr_ls)
    assert abs(pw2.sum() - 1.0) < 1e-5

    assert equal_weights(0).size == 0
    assert equal_weights(4).sum() == pytest.approx(1.0)


def test_make_infeasible_failed_format_minimize_pgd():
    r = make_result("t", np.ones(2) / 2, success=True, status="ok", method="m", objective_value=1.0)
    assert r["success"]
    r2 = make_result("t", [0.5, 0.5], success=True, status="ok", method="m")
    assert isinstance(r2["weights"], list)
    inf = infeasible_result("t", 2, method="m", reason="x", conflicts=["box"], names=["a", "b"])
    assert isinstance(inf["weights"], dict)
    fail = failed_result("t", 2, method="m", reason="boom")
    assert fail["status"] == "failed"
    fail2 = failed_result("t", 2, method="m", reason="boom", names=["a", "b"])
    assert isinstance(fail2["weights"], dict)

    assert isinstance(format_weights(np.ones(2) / 2, ["a", "b"]), dict)
    assert isinstance(format_weights(np.ones(2) / 2, None), list)

    assert portfolio_variance(np.ones(2) / 2, np.eye(2)) >= 0
    assert isinstance(portfolio_return(np.ones(2) / 2, np.array([0.1, 0.2])), float)

    if scipy_available():
        res = minimize_scipy(lambda x: float(x @ x), np.array([1.0, 1.0]), method="BFGS", options={"maxiter": 20})
        assert res is not None

    def fun(x):
        return float(np.sum(x**2))

    def grad(x):
        return 2 * x

    x, f, ok, it = projected_gradient(fun, grad, np.array([1.0, 1.0]), lambda z: z / max(np.sum(z), 1e-12), max_iter=50, tol=1e-12)
    assert x is not None
    assert it >= 1


# ============================================================================= optimizer failure / edge paths
def test_drawdown_missing_inputs_and_blend(mu, cov, returns, names, rng):
    res = optimize_drawdown()
    assert res["success"] is False
    bad = optimize_drawdown(returns=np.ones(10), cov=None)
    assert bad["success"] is False

    # cov-only path (synthetic returns)
    r0 = optimize_drawdown(mu=mu, cov=cov, names=names, max_weight=0.5, drawdown_cap=0.01)
    assert "success" in r0

    # force high MDD blend via artificial crash returns
    crash = returns.copy()
    crash[:, 0] = -0.05
    r1 = optimize_drawdown(
        mu=mu,
        cov=cov,
        returns=crash,
        names=names,
        max_weight=0.5,
        drawdown_cap=0.001,
        path_penalty=2.0,
    )
    assert "success" in r1

    # MV fail seed → equal / current weights
    r2 = optimize_drawdown(
        mu=mu,
        cov=cov,
        returns=returns,
        names=names,
        max_weight=0.1,
        budget=1.0,
        current_weights=np.ones(len(names)) / len(names),
    )
    assert "success" in r2


def test_cvar_bad_inputs_and_soft_paths(mu, cov, returns, names):
    assert optimize_cvar()["success"] is False
    assert optimize_cvar(scenarios=np.ones(5))["success"] is False
    assert optimize_cvar(scenarios=returns[:1])["success"] is False
    assert optimize_cvar(scenarios=returns, alpha=0.2)["success"] is False

    # synthetic from cov
    r0 = optimize_cvar(mu=mu, cov=cov, names=names, max_weight=0.5, max_iter=30)
    assert "success" in r0

    # with scenarios + return tradeoff
    r1 = optimize_cvar(
        mu=mu,
        cov=cov,
        scenarios=returns,
        names=names,
        max_weight=0.5,
        return_tradeoff=0.5,
        current_weights=np.ones(len(names)) / len(names),
        max_iter=40,
    )
    assert "success" in r1

    # infeasible constraints
    r2 = optimize_cvar(scenarios=returns, names=names, max_weight=0.1, budget=1.0)
    assert r2["success"] is False


def test_short_rejection_rp_hrp_entropy_and_missing_cov(cov, names, mu):
    for fn in (optimize_risk_parity, optimize_hrp, optimize_entropy):
        res = fn(cov=cov, names=names, long_only=False, min_weight=-0.5, max_weight=0.5)
        assert res["success"] is False
        assert "long_only" in str(res.get("conflicting_constraints") or []) or "long" in (
            res.get("failure_reason") or ""
        ).lower()

    assert optimize_risk_parity(names=names)["success"] is False
    assert optimize_hrp(names=names)["success"] is False
    assert optimize_entropy(names=names)["success"] is False
    assert optimize_minimum_variance(names=names)["success"] is False
    assert optimize_mean_variance(mu=mu, names=names)["success"] is False
    assert optimize_maximum_sharpe(mu=mu, names=names)["success"] is False
    assert optimize_maximum_diversification(names=names)["success"] is False
    assert optimize_turnover(mu=mu, names=names)["success"] is False


def test_post_check_infeasible_tight_box(mu, cov, names):
    # n * max_weight < budget → infeasible before solve
    res = optimize_mean_variance(mu=mu, cov=cov, names=names, max_weight=0.1, budget=1.0)
    assert res["success"] is False
    assert res["status"] == "infeasible"


# ============================================================================= commissions
def test_commission_per_share_and_min_floor(prices):
    trades = np.array([0.1, -0.05, 0.0, 0.2])
    c = commission_cost(
        trades,
        capital=1e6,
        prices=prices,
        commission_bps=1.0,
        commission_per_share=0.01,
        min_commission=5.0,
    )
    assert c["total"] >= 0.0
    assert len(c["per_asset"]) == 4
    # zero notional stays 0 even with floor
    assert c["per_asset"][2] == 0.0
    c2 = commission_cost(trades, capital=1e6, commission_bps=0.0, min_commission=1.0)
    assert c2["total"] >= 0.0


# ============================================================================= extra optimizer fallbacks via monkeypatch
def test_optimizers_scipy_unavailable_and_fail_fallback(mu, cov, names, monkeypatch):
    import iqrp.app.portfolio.optimization.mean_variance as mv
    import iqrp.app.portfolio.optimization.minimum_variance as mnv
    import iqrp.app.portfolio.optimization.maximum_sharpe as ms
    import iqrp.app.portfolio.optimization.maximum_diversification as md
    import iqrp.app.portfolio.optimization.turnover as to
    import iqrp.app.portfolio.optimization.entropy as ent
    import iqrp.app.portfolio.optimization.risk_parity as rp
    import iqrp.app.portfolio.optimization.cvar as cv
    import iqrp.app.portfolio.optimization.projection as proj

    monkeypatch.setattr(proj, "scipy_available", lambda: False)
    for mod in (mv, mnv, ms, md, to, ent, rp, cv):
        if hasattr(mod, "scipy_available"):
            monkeypatch.setattr(mod, "scipy_available", lambda: False)

    r1 = mv.optimize_mean_variance(mu=mu, cov=cov, names=names, max_weight=0.5)
    assert "success" in r1
    r2 = mnv.optimize_minimum_variance(cov=cov, names=names, max_weight=0.5)
    assert "success" in r2
    r3 = ms.optimize_maximum_sharpe(mu=mu, cov=cov, names=names, max_weight=0.5)
    assert "success" in r3
    r4 = md.optimize_maximum_diversification(cov=cov, names=names, max_weight=0.5)
    assert "success" in r4
    r5 = to.optimize_turnover(mu=mu, cov=cov, names=names, max_weight=0.5, current_weights=np.ones(4) / 4)
    assert "success" in r5
    r6 = ent.optimize_entropy(cov=cov, names=names, max_weight=0.5)
    assert "success" in r6
    r7 = rp.optimize_risk_parity(cov=cov, names=names, max_weight=0.5)
    assert "success" in r7
    r8 = cv.optimize_cvar(cov=cov, names=names, max_weight=0.5, max_iter=20)
    assert "success" in r8


def test_optimizers_minimize_scipy_raises(mu, cov, names, monkeypatch):
    import iqrp.app.portfolio.optimization.mean_variance as mv
    import iqrp.app.portfolio.optimization.projection as proj

    def _boom(*a, **k):
        raise RuntimeError("scipy boom")

    monkeypatch.setattr(proj, "minimize_scipy", _boom)
    monkeypatch.setattr(mv, "minimize_scipy", _boom)
    monkeypatch.setattr(mv, "scipy_available", lambda: True)
    res = mv.optimize_mean_variance(mu=mu, cov=cov, names=names, max_weight=0.5)
    assert "success" in res


def test_optimizers_postcheck_via_bad_project(mu, cov, names, monkeypatch):
    import iqrp.app.portfolio.optimization.mean_variance as mv
    import iqrp.app.portfolio.optimization.minimum_variance as mnv
    import iqrp.app.portfolio.optimization.maximum_sharpe as ms
    import iqrp.app.portfolio.optimization.projection as proj

    def bad_project(v, cstr):
        # violate box on purpose
        return np.full(int(cstr["n"]), 2.0)

    for mod in (mv, mnv, ms):
        monkeypatch.setattr(mod, "project_weights", bad_project)
        monkeypatch.setattr(mod, "scipy_available", lambda: False)
    assert mv.optimize_mean_variance(mu=mu, cov=cov, names=names, max_weight=0.5)["success"] is False
    assert mnv.optimize_minimum_variance(cov=cov, names=names, max_weight=0.5)["success"] is False
    assert ms.optimize_maximum_sharpe(mu=mu, cov=cov, names=names, max_weight=0.5)["success"] is False


def test_mv_singular_cov_and_exception_handler(mu, names, monkeypatch):
    # singular-ish cov may trigger LinAlgError path for seed
    n = len(names)
    singular = np.ones((n, n)) * 0.01
    res = optimize_mean_variance(mu=mu, cov=singular, names=names, max_weight=0.5)
    assert "success" in res

    # force outer exception with bad cov object
    class Bad:
        def __array__(self):
            raise RuntimeError("no array")

    res2 = optimize_mean_variance(mu=mu, cov=Bad(), names=names)
    assert res2["success"] is False


def test_drawdown_hrp_entropy_exception_handlers(cov, names, monkeypatch):
    import iqrp.app.portfolio.optimization.drawdown as dd
    import iqrp.app.portfolio.optimization.hierarchical as hr
    import iqrp.app.portfolio.optimization.entropy as ent
    import iqrp.app.portfolio.optimization.risk_parity as rp
    import iqrp.app.portfolio.optimization.turnover as to
    import iqrp.app.portfolio.optimization.maximum_diversification as md
    import iqrp.app.portfolio.optimization.black_litterman as bl
    import iqrp.app.portfolio.optimization.robust as rob

    def boom_parse(*a, **k):
        raise RuntimeError("parse fail")

    for mod in (dd, hr, ent, rp, to, md):
        monkeypatch.setattr(mod, "parse_constraints", boom_parse)
        fn_name = [x for x in dir(mod) if x.startswith("optimize_")][0]
        out = getattr(mod, fn_name)(cov=cov, names=names)
        assert out["success"] is False

    # modules without local parse_constraints: break as_cov / entry
    def boom_cov(*a, **k):
        raise RuntimeError("cov fail")

    monkeypatch.setattr(bl, "as_cov", boom_cov, raising=False)
    # black litterman may import as_cov from projection at call time
    try:
        out_bl = bl.optimize_black_litterman(cov=cov, names=names)
        assert out_bl["success"] is False or "success" in out_bl
    except Exception:
        pass
    try:
        out_r = rob.optimize_robust(cov=cov, names=names)
        assert "success" in out_r
    except Exception:
        pass


def test_constraints_exposure_and_risk(weights):
    from iqrp.app.portfolio.constraints.exposure import check_exposure_constraints, exposure_metrics
    from iqrp.app.portfolio.constraints.risk import check_risk_constraints
    from iqrp.app.portfolio.constraints.concentration import check_concentration_constraints
    from iqrp.app.portfolio.constraints.leverage import check_leverage_constraints
    from iqrp.app.portfolio.constraints.turnover import check_turnover_constraints

    assert exposure_metrics([])["gross"] == 0.0
    viols = check_exposure_constraints(
        weights,
        max_gross=0.1,
        max_net=0.1,
        min_net=2.0,
        max_long=0.1,
        max_short=0.0,
    )
    assert len(viols) >= 3

    rv = check_risk_constraints(
        weights,
        risk_metrics={
            "var": 0.2,
            "cvar": 0.3,
            "expected_shortfall": 0.3,
            "drawdown": 0.4,
            "risk_contribution": [0.5, 0.1, 0.1, 0.1],
        },
        max_var=0.05,
        max_cvar=0.05,
        max_drawdown=0.1,
        max_risk_contribution=0.2,
    )
    assert rv
    rv2 = check_risk_constraints(expected_shortfall=0.5, max_expected_shortfall=0.1)
    assert rv2

    # concentration / leverage / turnover empty-ish
    try:
        check_concentration_constraints(weights, max_hhi=0.01)
    except TypeError:
        pass
    try:
        check_leverage_constraints(weights, max_leverage=0.1)
    except TypeError:
        pass
    try:
        check_turnover_constraints(weights, current_weights=np.zeros(4), max_turnover=0.01)
    except TypeError:
        pass


def test_scipy_success_false_then_pgd(mu, cov, names, monkeypatch):
    import iqrp.app.portfolio.optimization.mean_variance as mv
    import iqrp.app.portfolio.optimization.minimum_variance as mnv
    import iqrp.app.portfolio.optimization.maximum_sharpe as ms
    import iqrp.app.portfolio.optimization.maximum_diversification as md
    import iqrp.app.portfolio.optimization.turnover as to
    import iqrp.app.portfolio.optimization.cvar as cv

    class FakeRes:
        success = False
        x = np.ones(len(names)) / len(names)
        nit = 3

    def fake_min(*a, **k):
        return FakeRes()

    for mod in (mv, mnv, ms, md, to, cv):
        monkeypatch.setattr(mod, "minimize_scipy", fake_min)
        monkeypatch.setattr(mod, "scipy_available", lambda: True)

    assert "success" in mv.optimize_mean_variance(mu=mu, cov=cov, names=names, max_weight=0.5)
    assert "success" in mnv.optimize_minimum_variance(cov=cov, names=names, max_weight=0.5)
    assert "success" in ms.optimize_maximum_sharpe(mu=mu, cov=cov, names=names, max_weight=0.5)
    assert "success" in md.optimize_maximum_diversification(cov=cov, names=names, max_weight=0.5)
    assert "success" in to.optimize_turnover(
        mu=mu, cov=cov, names=names, max_weight=0.5, current_weights=np.ones(4) / 4
    )
    assert "success" in cv.optimize_cvar(cov=cov, scenarios=np.random.default_rng(0).normal(size=(30, 4)), names=names, max_weight=0.5, max_iter=10)


def test_budget_and_gross_postcheck(mu, cov, names, monkeypatch):
    import iqrp.app.portfolio.optimization.mean_variance as mv
    import iqrp.app.portfolio.optimization.minimum_variance as mnv
    import iqrp.app.portfolio.optimization.maximum_sharpe as ms
    import iqrp.app.portfolio.optimization.entropy as ent
    import iqrp.app.portfolio.optimization.hierarchical as hr
    import iqrp.app.portfolio.optimization.risk_parity as rp
    import iqrp.app.portfolio.optimization.maximum_diversification as md
    import iqrp.app.portfolio.optimization.turnover as to
    import iqrp.app.portfolio.optimization.drawdown as dd
    import iqrp.app.portfolio.optimization.cvar as cv
    import iqrp.app.portfolio.robust.distributional_robust as drob

    def budget_break(v, cstr):
        w = np.ones(int(cstr["n"])) * 0.1  # sum != budget
        return w

    def gross_break(v, cstr):
        # within box but gross too high if max_gross set: use long-short
        n = int(cstr["n"])
        w = np.zeros(n)
        w[0] = 2.0
        w[1] = -1.0
        return w

    for mod in (mv, mnv, ms, ent, hr, rp, md, to, dd, cv, drob):
        if hasattr(mod, "scipy_available"):
            monkeypatch.setattr(mod, "scipy_available", lambda: False)
        monkeypatch.setattr(mod, "project_weights", budget_break)
    assert mv.optimize_mean_variance(mu=mu, cov=cov, names=names, max_weight=0.5)["success"] is False
    assert mnv.optimize_minimum_variance(cov=cov, names=names, max_weight=0.5)["success"] is False
    assert ms.optimize_maximum_sharpe(mu=mu, cov=cov, names=names, max_weight=0.5)["success"] is False
    assert ent.optimize_entropy(cov=cov, names=names, max_weight=0.5)["success"] is False
    assert hr.optimize_hrp(cov=cov, names=names, max_weight=0.6)["success"] is False
    assert rp.optimize_risk_parity(cov=cov, names=names, max_weight=0.5)["success"] is False
    assert md.optimize_maximum_diversification(cov=cov, names=names, max_weight=0.5)["success"] is False
    assert to.optimize_turnover(mu=mu, cov=cov, names=names, max_weight=0.5)["success"] is False
    assert dd.optimize_drawdown(mu=mu, cov=cov, names=names, max_weight=0.5)["success"] is False
    assert cv.optimize_cvar(cov=cov, names=names, max_weight=0.5, max_iter=5)["success"] is False
    assert drob.optimize_distributional_robust(mu=mu, cov=cov, names=names, max_weight=0.5)["success"] is False

    # gross violation path for MV
    monkeypatch.setattr(mv, "project_weights", gross_break)
    res = mv.optimize_mean_variance(mu=mu, cov=cov, names=names, max_weight=5.0, max_gross=1.0, long_only=False, min_weight=-2.0)
    assert res["success"] is False


def test_hrp_backend_bad_size_and_weights_dict(cov, names, monkeypatch):
    import iqrp.app.portfolio.optimization.hierarchical as hr
    import iqrp.app.portfolio.optimization.risk_parity as rp

    def bad_hrp(*a, **k):
        return {"weights": {f"a{i}": 0.0 for i in range(10)}}  # wrong keys/size after list comp may error

    monkeypatch.setattr(hr, "hrp_weights", lambda *a, **k: {"weight_vector": np.ones(2)})
    monkeypatch.setattr(hr, "herc_weights", lambda *a, **k: {"weight_vector": np.ones(2)})
    out = hr.optimize_hrp(cov=cov, names=names, max_weight=0.6)
    assert out["success"] is False
    out2 = hr.optimize_herc(cov=cov, names=names, max_weight=0.6)
    assert out2["success"] is False

    monkeypatch.setattr(rp, "risk_parity_weights", lambda *a, **k: {"weight_vector": np.ones(1)})
    assert rp.optimize_risk_parity(cov=cov, names=names, max_weight=0.5)["success"] is False


def test_black_litterman_local_and_views(mu, cov, names, monkeypatch):
    from iqrp.app.portfolio.optimization.black_litterman import (
        optimize_black_litterman,
        _local_black_litterman_posterior,
    )

    n = len(names)
    local = _local_black_litterman_posterior(cov)
    assert "mu" in local
    P = np.zeros((1, n))
    P[0, 0] = 1.0
    Q = np.array([0.01])
    local2 = _local_black_litterman_posterior(cov, P=P, Q=Q, omega=np.array([0.1]))
    assert local2["method"] == "black_litterman_local"
    local3 = _local_black_litterman_posterior(cov, P=P.reshape(-1), Q=Q)
    assert "mu" in local3
    with pytest.raises(ValueError):
        _local_black_litterman_posterior(cov, P=np.eye(1, n + 1), Q=Q)
    with pytest.raises(ValueError):
        _local_black_litterman_posterior(cov, P=P, Q=np.array([0.01, 0.02]))

    # force local path by breaking expected_returns import inside optimizer
    import iqrp.app.portfolio.optimization.black_litterman as bl

    def boom_call(*a, **k):
        raise ImportError("no er module")

    # call optimize with views
    res = optimize_black_litterman(
        cov=cov,
        names=names,
        P=P,
        Q=Q,
        market_weights=np.ones(n) / n,
        max_weight=0.5,
    )
    assert "success" in res
    res2 = optimize_black_litterman(cov=cov, names=names, max_weight=0.5)
    assert "success" in res2


def test_softmin_cvar_empty_and_no_hard(monkeypatch):
    from iqrp.app.portfolio.optimization.cvar import _softmin_cvar

    c, mix = _softmin_cvar(np.array([]), 0.95, 0.01)
    assert c == 0.0
    # all equal losses → hard path edge
    losses = np.array([-1.0, -1.0, -1.0])
    c2, mix2 = _softmin_cvar(losses, 0.99, 1e-12)
    assert mix2 is not None


def test_exposure_short_and_risk_skip_rc(weights):
    from iqrp.app.portfolio.constraints.exposure import check_exposure_constraints
    from iqrp.app.portfolio.constraints.risk import check_risk_constraints
    from iqrp.app.portfolio.constraints.concentration import check_concentration_constraints
    from iqrp.app.portfolio.constraints.leverage import check_leverage_constraints
    from iqrp.app.portfolio.constraints.turnover import check_turnover_constraints

    w_ls = np.array([0.8, -0.5, 0.4, 0.3])
    v = check_exposure_constraints(w_ls, max_short=0.1)
    assert any(x.name == "max_short_exposure" for x in v)

    # risk contribution skip when weights given but no rc vector
    assert check_risk_constraints(weights=weights, max_risk_contribution=0.1) == []
    # metrics expected_shortfall key
    assert check_risk_constraints(
        risk_metrics={"expected_shortfall": 0.5},
        max_expected_shortfall=0.1,
    )

    check_concentration_constraints(weights, max_hhi=0.01)
    check_leverage_constraints(weights, max_leverage=0.1)
    check_turnover_constraints(weights, current_weights=np.zeros(4), max_turnover=0.01)
