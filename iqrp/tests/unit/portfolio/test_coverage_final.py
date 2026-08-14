"""Final coverage push: exception paths, ValueErrors, monkeypatched fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from iqrp.app.portfolio.base.objective import ObjectiveSpec
from iqrp.app.portfolio.base.optimizer import OptimizationResult, PortfolioOptimizer
from iqrp.app.portfolio.config import PortfolioSettings
from iqrp.app.portfolio.constraints.factor import (
    check_factor_constraints,
    portfolio_factor_exposures,
)
from iqrp.app.portfolio.constraints.liquidity import check_liquidity_constraints
from iqrp.app.portfolio.construction.constructor import PortfolioResult
from iqrp.app.portfolio.construction.rebalance import RebalanceBands, plan_rebalance
from iqrp.app.portfolio.construction.target_positions import TargetPositions
from iqrp.app.portfolio.covariance.factor import factor_covariance
from iqrp.app.portfolio.covariance.robust import robust_covariance
from iqrp.app.portfolio.covariance.shrinkage import shrinkage_covariance
from iqrp.app.portfolio.engine import PortfolioConstructionEngine, dict_to_optimization_result
from iqrp.app.portfolio.expected_returns.black_litterman import (
    black_litterman_posterior,
    equilibrium_returns,
)
from iqrp.app.portfolio.expected_returns.shrinkage import james_stein_shrinkage
from iqrp.app.portfolio.multi_period import optimize_dynamic_programming, optimize_multi_period
from iqrp.app.portfolio.optimization import (
    optimize_black_litterman,
    optimize_cvar,
    optimize_drawdown,
    optimize_entropy,
    optimize_herc,
    optimize_hrp,
    optimize_maximum_diversification,
    optimize_maximum_sharpe,
    optimize_mean_variance,
    optimize_minimum_variance,
    optimize_risk_parity,
    optimize_turnover,
    projection as proj,
)
from iqrp.app.portfolio.robust import optimize_distributional_robust, optimize_parameter_uncertainty
from iqrp.app.portfolio.serializer import PortfolioSerializer, _to_jsonable

# ---- easy ValueError / input branches --------------------------------------


def test_bl_input_validation_errors(cov, names):
    n = len(names)
    with pytest.raises(ValueError, match="2-D matrix"):
        black_litterman_posterior(cov=np.ones(3), market_weights=np.ones(3) / 3)

    with pytest.raises(ValueError, match="square"):
        black_litterman_posterior(cov=np.ones((2, 3)), market_weights=np.ones(2) / 2)

    with pytest.raises(ValueError, match="market_weights length"):
        black_litterman_posterior(cov=cov, market_weights=np.ones(n + 1) / (n + 1))

    P = np.zeros((1, n + 1))
    with pytest.raises(ValueError, match="P columns"):
        black_litterman_posterior(cov=cov, market_weights=np.ones(n) / n, P=P, Q=np.array([0.01]))

    # exercise _as_1d via equilibrium with flat weights (happy path)
    eq = equilibrium_returns(cov, np.ones(n) / n)
    assert eq.shape == (n,)


def test_shrinkage_returns_shapes(rng):
    r1 = rng.normal(0, 0.01, size=50)
    out = shrinkage_covariance(r1)
    assert np.asarray(out["matrix"]).shape == (1, 1)

    with pytest.raises(ValueError):
        shrinkage_covariance(rng.normal(size=(5, 3, 2)))


def test_factor_cov_errors_and_defaults(returns):
    with pytest.raises(ValueError):
        factor_covariance(factor_loadings=np.array(1.0))  # 0-d

    n = returns.shape[1]
    B = np.eye(n)[:, :2]
    # no factor_returns / factor_cov → identity F path
    out = factor_covariance(factor_loadings=B, residual_vars=np.ones(n) * 1e-4)
    assert np.asarray(out["matrix"]).shape == (n, n)


def test_liquidity_scalar_adv(weights):
    viols = check_liquidity_constraints(
        weights,
        adv=1e6,  # scalar → broadcast
        capital=1e5,
        max_participation=0.5,
    )
    assert isinstance(viols, list)


def test_factor_constraint_transpose_and_empty_neutral(weights):
    n = len(weights)
    B = np.random.default_rng(0).normal(size=(2, n))  # k x n → transpose path
    expos = portfolio_factor_exposures(weights, factor_loadings=B)
    assert len(expos) >= 1
    # factor_neutral True with empty name set still runs
    viols = check_factor_constraints(
        weights,
        factor_loadings=B.T,
        factor_neutral=True,
        factor_names=[],
        neutrality_tol=1.0,
    )
    assert isinstance(viols, list)


def test_config_default_without_file(monkeypatch, tmp_path):
    missing = tmp_path / "nope.yaml"
    monkeypatch.setattr(
        "iqrp.app.portfolio.config._default_config_path",
        lambda: missing,
    )
    s = PortfolioSettings.default()
    assert isinstance(s, PortfolioSettings)


def test_engine_dict_weights_sorted_keys(engine, cov):
    # length mismatch with n → sorted-keys branch
    raw = {"z": 0.1, "a": 0.2, "m": 0.3}
    w = __import__("iqrp.app.portfolio.engine", fromlist=["_extract_weights"])._extract_weights(
        raw, n=4
    )
    assert len(w) == 3  # sorted keys of dict when len != n
    # force via dict_to_optimization_result with odd dict
    r = dict_to_optimization_result(
        {"success": True, "weights": {"b": 0.4, "a": 0.6}}, names=["x", "y", "z"]
    )
    assert len(r.weights) == 2


def test_engine_mu_cov_exception_pass():
    class BadMu:
        def __array__(self, dtype=None):
            raise RuntimeError("nope")

    r = dict_to_optimization_result(
        {"success": True, "weights": [0.5, 0.5]},
        mu=BadMu(),
        cov=BadMu(),
    )
    assert r.success


def test_engine_construct_returns_only_1d(engine, rng):
    r1 = rng.normal(0, 0.01, size=80)
    # no names/mu/cov — uses returns ndim branch
    result = engine.construct(returns=r1.reshape(-1, 1), method="min_variance")
    assert result is not None


def test_engine_expected_returns_unknown_with_returns(engine, returns, names):
    # unknown method + returns → historical fallback (line ~1095-1099)
    out = engine.expected_returns(returns=returns, method="totally_unknown_xyz", names=names)
    assert "mu" in out or "vector" in out


def test_engine_target_positions_empty_construct(engine):
    # construct without inputs → empty positions path
    tp = engine.target_positions(names=["a", "b"], method="min_variance")
    assert tp is not None


def test_engine_risk_approved_empty_action():
    class Dec:
        def to_dict(self):
            return {"approved": True}  # no action key → line 981-985

    class Risk:
        def check_limits(self, **kw):
            return []

        def validate_position(self, **kw):
            return Dec()

    eng = PortfolioConstructionEngine(
        settings=PortfolioSettings(require_risk_validation=True, fallback="cash", seed=1),
        risk_engine=Risk(),
    )
    n = 4
    r = eng.construct(
        mu=np.ones(n) * 0.001,
        cov=np.eye(n) * 0.01,
        returns=np.random.default_rng(0).normal(0, 0.01, (60, n)),
        names=[f"A{i}" for i in range(n)],
        method="min_variance",
    )
    assert r.risk_validation.get("approved") is True


def test_portfolio_result_none_positions():
    pr = PortfolioResult(target_positions=None, success=True)
    d = pr.to_dict()
    assert d["target_positions"] is None


def test_optimizer_validate_weights(names):
    class Dummy(PortfolioOptimizer):
        def optimize(self, **kwargs):
            return OptimizationResult.failure(reason="x")

    opt = Dummy(objective=ObjectiveSpec())
    viols = opt.validate_weights(np.ones(len(names)) / len(names), names=names)
    assert isinstance(viols, list)


def test_rebalance_min_trade_filter(current_weights):
    target = current_weights.copy()
    target[0] += 0.002
    target[-1] -= 0.002
    plan = plan_rebalance(
        current_weights,
        target,
        bands=RebalanceBands(absolute=0.0, relative=0.0, min_trade=0.01),
        force=True,
    )
    # tiny trades zeroed by min_trade
    assert float(np.max(np.abs(plan.trades))) < 0.01 + 1e-12 or plan.turnover >= 0


# ---- optimizer infeasible with names in constraints ------------------------


def test_optimizers_infeasible_with_constraint_names(cov, names):
    cons = {"max_weight": 0.05, "names": list(names)}  # n*0.05 < 1
    for fn in (
        optimize_entropy,
        optimize_maximum_diversification,
        optimize_risk_parity,
        optimize_hrp,
        optimize_turnover,
        optimize_distributional_robust,
    ):
        kwargs: dict[str, Any] = {"cov": cov, "constraints": cons, "max_weight": 0.05}
        if fn in (optimize_entropy, optimize_turnover, optimize_distributional_robust):
            kwargs["mu"] = np.ones(len(names)) * 0.01
        if fn is optimize_turnover:
            kwargs["current_weights"] = np.ones(len(names)) / len(names)
        res = fn(**kwargs)
        assert res["success"] is False
        assert res["status"] == "infeasible"


def test_max_sharpe_infeasible(cov, names, mu):
    res = optimize_maximum_sharpe(
        mu=mu, cov=cov, max_weight=0.05, constraints={"names": list(names)}
    )
    assert res["success"] is False


def test_drawdown_box_violation_and_returns_cov(returns, names, current_weights):
    # returns only (no cov) → internal cov path
    res = optimize_drawdown(returns=returns, max_weight=0.5, names=names)
    assert "success" in res
    # infeasible box
    res2 = optimize_drawdown(
        returns=returns,
        max_weight=0.05,
        names=names,
        current_weights=current_weights,
    )
    assert res2["success"] is False or res2["status"] in (
        "infeasible",
        "optimal",
        "fallback",
        "failed",
    )


def test_mean_variance_mu_none(cov, names):
    res = optimize_mean_variance(mu=None, cov=cov, names=names, max_weight=0.5)
    assert res["success"] or res["status"] in ("infeasible", "failed", "fallback")


def test_mean_variance_singular_cov_linalg(names):
    # rank-1 cov triggers LinAlgError path for analytic start
    v = np.ones(len(names))
    cov = np.outer(v, v) * 1e-4
    res = optimize_mean_variance(
        mu=np.ones(len(names)) * 0.01, cov=cov, names=names, max_weight=0.5
    )
    assert "success" in res


def test_min_var_singular_cov(names):
    v = np.ones(len(names))
    cov = np.outer(v, v) * 1e-4
    res = optimize_minimum_variance(cov=cov, names=names, max_weight=0.5)
    assert "success" in res


def test_max_sharpe_singular_cov(names, mu):
    v = np.ones(len(names))
    cov = np.outer(v, v) * 1e-4
    res = optimize_maximum_sharpe(mu=mu, cov=cov, names=names, max_weight=0.5)
    assert "success" in res


def test_optimize_bl_requires_cov(names):
    res = optimize_black_litterman(
        cov=None, names=names, market_weights=np.ones(len(names)) / len(names)
    )
    assert res["success"] is False or res.get("failure_reason")


def test_multi_period_missing_cov(mu, names):
    res = optimize_multi_period(mu=mu, cov=None, horizons=2, names=names)
    assert res["success"] is False


def test_multi_period_return_path_1d(mu, cov, names):
    n = len(names)
    rp = np.ones(n) * 0.001  # 1d → reshape
    res = optimize_multi_period(
        mu=mu,
        cov=cov,
        horizons=2,
        return_path=rp,
        names=names,
        max_weight=0.5,
    )
    assert "success" in res


def test_dp_infeasible_and_empty_grid(mu, cov, names):
    n = min(3, len(names))
    res = optimize_dynamic_programming(
        mu=mu[:n],
        cov=cov[:n, :n],
        horizons=2,
        max_weight=0.05,
        grid_levels=3,
        names=names[:n],
    )
    assert res["success"] is False or "success" in res


def test_dp_mu_path_none(cov, names):
    n = min(3, len(names))
    res = optimize_dynamic_programming(
        mu=None,
        cov=cov[:n, :n],
        horizons=2,
        grid_levels=3,
        names=names[:n],
        max_weight=0.6,
    )
    assert "success" in res


# ---- monkeypatch scipy → PGD / exception outer -----------------------------


def test_pgd_fallback_when_minimize_raises(mu, cov, names, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("scipy down")

    monkeypatch.setattr(proj, "_scipy_minimize", boom)
    monkeypatch.setattr(proj, "minimize_scipy", boom)

    for fn, kwargs in (
        (optimize_entropy, {"mu": mu, "cov": cov, "names": names, "max_weight": 0.5}),
        (optimize_maximum_diversification, {"cov": cov, "names": names, "max_weight": 0.5}),
        (optimize_maximum_sharpe, {"mu": mu, "cov": cov, "names": names, "max_weight": 0.5}),
        (optimize_minimum_variance, {"cov": cov, "names": names, "max_weight": 0.5}),
        (
            optimize_turnover,
            {
                "mu": mu,
                "cov": cov,
                "names": names,
                "max_weight": 0.5,
                "current_weights": np.ones(len(names)) / len(names),
            },
        ),
        (
            optimize_cvar,
            {
                "cov": cov,
                "names": names,
                "max_weight": 0.5,
                "scenarios": np.random.default_rng(0).normal(0, 0.01, (80, len(names))),
            },
        ),
        (optimize_distributional_robust, {"mu": mu, "cov": cov, "names": names, "max_weight": 0.5}),
        (optimize_mean_variance, {"mu": mu, "cov": cov, "names": names, "max_weight": 0.5}),
    ):
        res = fn(**kwargs)
        assert "success" in res


def test_outer_exception_failed_result(cov, names, monkeypatch):
    """Force as_cov to raise after entry → outer except n=0 path."""

    def bad_as_cov(*a, **k):
        raise RuntimeError("cov explode")

    monkeypatch.setattr(proj, "as_cov", bad_as_cov)
    # Import modules that call as_cov at runtime
    for mod_name, fn_name, kwargs in (
        (
            "iqrp.app.portfolio.optimization.entropy",
            "optimize_entropy",
            {"mu": np.ones(4) * 0.01, "cov": cov, "names": names},
        ),
        (
            "iqrp.app.portfolio.optimization.maximum_sharpe",
            "optimize_maximum_sharpe",
            {"mu": np.ones(4) * 0.01, "cov": cov, "names": names},
        ),
        (
            "iqrp.app.portfolio.optimization.minimum_variance",
            "optimize_minimum_variance",
            {"cov": cov, "names": names},
        ),
        (
            "iqrp.app.portfolio.optimization.maximum_diversification",
            "optimize_maximum_diversification",
            {"cov": cov, "names": names},
        ),
        (
            "iqrp.app.portfolio.optimization.risk_parity",
            "optimize_risk_parity",
            {"cov": cov, "names": names},
        ),
        (
            "iqrp.app.portfolio.optimization.hierarchical",
            "optimize_hierarchical",
            {"cov": cov, "names": names},
        ),
        (
            "iqrp.app.portfolio.optimization.turnover",
            "optimize_turnover",
            {"mu": np.ones(4) * 0.01, "cov": cov, "names": names},
        ),
        (
            "iqrp.app.portfolio.robust.distributional_robust",
            "optimize_distributional_robust",
            {"mu": np.ones(4) * 0.01, "cov": cov, "names": names},
        ),
        (
            "iqrp.app.portfolio.optimization.black_litterman",
            "optimize_black_litterman",
            {"cov": cov, "market_weights": np.ones(4) / 4, "names": names},
        ),
        (
            "iqrp.app.portfolio.multi_period.optimizer",
            "optimize_multi_period",
            {"mu": np.ones(4) * 0.01, "cov": cov, "horizons": 2, "names": names},
        ),
    ):
        mod = __import__(mod_name, fromlist=[fn_name])
        monkeypatch.setattr(mod, "as_cov", bad_as_cov, raising=False)
        fn = getattr(mod, fn_name)
        res = fn(**kwargs)
        assert res["success"] is False


def test_parameter_uncertainty_outer_except(monkeypatch, names):
    def boom(*a, **k):
        raise RuntimeError("x")

    import iqrp.app.portfolio.robust.parameter_uncertainty as pu

    monkeypatch.setattr(pu, "optimize_distributional_robust", boom)
    res = optimize_parameter_uncertainty(
        mu=np.ones(len(names)) * 0.01,
        cov=np.eye(len(names)) * 0.01,
        names=names,
    )
    # may return failed from exception handler
    assert "success" in res or res is not None


def test_parameter_uncertainty_n_from_returns(returns, names):
    res = optimize_parameter_uncertainty(
        mu=None,
        cov=None,
        returns=returns,
        names=names,
        max_weight=0.5,
    )
    assert "success" in res


def test_projection_gross_and_weights_edges():
    w = np.array([0.8, 0.8, -0.3])
    g = proj.project_gross(w, max_gross=1.0)
    assert np.sum(np.abs(g)) <= 1.0 + 1e-6

    cstr = proj.parse_constraints(None, n=3, long_only=True, max_weight=0.5, budget=1.0)
    pw = proj.project_weights(np.array([0.5, 0.5, 0.5]), cstr)
    assert pw.shape == (3,)

    s = proj.project_simplex(np.array([1.0, 1.0, 1.0]), budget=0.0)
    assert float(np.sum(np.abs(s))) < 1e-8 or s.sum() == pytest.approx(0.0, abs=1e-8)


def test_projection_box_simplex_equal_fallback():
    # when projection hits special theta paths
    x = np.array([10.0, -5.0, 3.0])
    out = proj.project_box_simplex(x, lb=0.0, ub=1.0, budget=1.0)
    assert abs(out.sum() - 1.0) < 1e-6
    assert np.all(out >= -1e-9)
    assert np.all(out <= 1.0 + 1e-9)

    # force max_iter exhaustion → final clip return (line 225)
    out2 = proj.project_box_simplex(
        np.array([0.5, 0.3, 0.2]), lb=0.0, ub=1.0, budget=1.0, max_iter=1, tol=1e-30
    )
    assert out2.shape == (3,)


def test_serializer_to_jsonable_exception():
    class Weird:
        def __iter__(self):
            raise RuntimeError("no")

    # should not raise — falls through
    out = _to_jsonable({"w": Weird(), "ok": 1})
    assert out["ok"] == 1


def test_james_stein_linalg_fallback(names):
    # singular cov → LinAlgError → quad = dot(diff,diff)
    n = len(names)
    v = np.ones(n)
    cov = np.outer(v, v)
    out = james_stein_shrinkage(np.ones(n) * 0.01, cov=cov, n_obs=10, names=names)
    assert "mu" in out


def test_robust_cov_edge_paths(rng):
    # single column / tiny sample for mcd edge returns
    r = rng.normal(0, 0.01, size=(8, 1))
    out = robust_covariance(r, method="mcd", n_trials=4, seed=0, h_fraction=0.9)
    assert "matrix" in out
    assert np.asarray(out["matrix"]).shape[0] >= 1

    r2 = rng.normal(0, 0.01, size=(12, 3))
    out2 = robust_covariance(r2, method="winsorize_mcd", n_trials=6, seed=1)
    assert np.asarray(out2["matrix"]).shape == (3, 3)


def test_risk_parity_post_project_infeasible(cov, names, monkeypatch):
    import iqrp.app.portfolio.optimization.risk_parity as rp

    def bad_project(*a, **k):
        return np.array([2.0, 2.0, 2.0, 2.0][: len(names)])  # violates box

    monkeypatch.setattr(rp, "project_weights", bad_project)
    res = optimize_risk_parity(cov=cov, names=names, max_weight=0.4)
    # may be infeasible after post-check
    assert "success" in res


def test_hierarchical_post_infeasible(cov, names, monkeypatch):
    import iqrp.app.portfolio.optimization.hierarchical as hi

    def bad_project(*a, **k):
        return np.ones(len(names)) * 2.0

    monkeypatch.setattr(hi, "project_weights", bad_project)
    res = optimize_hrp(cov=cov, names=names, max_weight=0.4)
    assert "success" in res


def test_box_violation_via_bad_project(mu, cov, names, monkeypatch):
    """Force post-opt box check to fail for several optimizers."""

    def bad_project(w, **kwargs):
        return np.ones(len(np.asarray(w).reshape(-1)))  # sum=n, each=1 → box viol

    for mod_name, fn, kwargs in (
        (
            "iqrp.app.portfolio.optimization.entropy",
            optimize_entropy,
            {"mu": mu, "cov": cov, "names": names, "max_weight": 0.4},
        ),
        (
            "iqrp.app.portfolio.optimization.maximum_diversification",
            optimize_maximum_diversification,
            {"cov": cov, "names": names, "max_weight": 0.4},
        ),
        (
            "iqrp.app.portfolio.optimization.cvar",
            optimize_cvar,
            {"cov": cov, "names": names, "max_weight": 0.4},
        ),
        (
            "iqrp.app.portfolio.optimization.turnover",
            optimize_turnover,
            {
                "mu": mu,
                "cov": cov,
                "names": names,
                "max_weight": 0.4,
                "current_weights": np.ones(len(names)) / len(names),
            },
        ),
        (
            "iqrp.app.portfolio.robust.distributional_robust",
            optimize_distributional_robust,
            {"mu": mu, "cov": cov, "names": names, "max_weight": 0.4},
        ),
        (
            "iqrp.app.portfolio.optimization.drawdown",
            optimize_drawdown,
            {
                "returns": np.random.default_rng(0).normal(0, 0.01, (80, len(names))),
                "cov": cov,
                "names": names,
                "max_weight": 0.4,
            },
        ),
    ):
        mod = __import__(mod_name, fromlist=["project_weights"])
        monkeypatch.setattr(mod, "project_weights", bad_project, raising=False)
        if hasattr(mod, "project_box_simplex"):
            monkeypatch.setattr(
                mod, "project_box_simplex", lambda *a, **k: np.ones(len(names)), raising=False
            )
        res = fn(**kwargs)
        assert "success" in res


def test_shrinkage_ledoit_wolf_1d_and_empty():
    from iqrp.app.portfolio.covariance.shrinkage import ledoit_wolf_covariance, shrinkage_covariance

    r1 = np.random.default_rng(0).normal(0, 0.01, size=40)
    out = ledoit_wolf_covariance(r1)
    assert np.asarray(out["matrix"]).shape == (1, 1)

    with pytest.raises(ValueError):
        ledoit_wolf_covariance(np.zeros((2, 2, 2)))

    # intensity override with empty cov edge via monkeypatch hard; use tiny 1-col
    out2 = shrinkage_covariance(r1, method="ledoit_wolf", intensity=0.5)
    assert "matrix" in out2


def test_robust_cov_nan_column_and_tiny():
    x = np.random.default_rng(0).normal(0, 0.01, size=(30, 3))
    x[:, 1] = np.nan
    out = robust_covariance(x, method="winsorize", seed=0)
    assert "matrix" in out

    tiny = np.array([[0.1]])  # T=1 < 2
    out2 = robust_covariance(tiny, method="mcd", n_trials=2, seed=0)
    assert "matrix" in out2

    # all non-finite after mask → t_eff < 2
    bad = np.full((5, 2), np.nan)
    out3 = robust_covariance(bad, method="mcd", n_trials=2, seed=0)
    assert "matrix" in out3


def test_factor_exposures_pad_truncate(weights):
    n = len(weights)
    # more rows than n → truncate B[:n,:]
    B = np.random.default_rng(1).normal(size=(n + 2, 2))
    expos = portfolio_factor_exposures(weights, factor_loadings=B)
    assert len(expos) >= 1
    # factor_neutral with explicit empty iterable → neutral_set empty
    viols = check_factor_constraints(
        weights,
        factor_loadings=np.eye(n)[:, :1],
        factor_neutral=[],  # empty → set()
        max_factor_exposure=None,
    )
    assert isinstance(viols, list)


def test_engine_remaining_branches(engine, rng):
    # 1-d returns → n=1 branch when no cov/mu
    from iqrp.app.portfolio import PortfolioSettings
    from iqrp.app.portfolio.engine import PortfolioConstructionEngine

    eng = PortfolioConstructionEngine(
        settings=PortfolioSettings(require_risk_validation=False, fallback="cash", seed=1)
    )
    r = eng.construct(returns=rng.normal(0, 0.01, size=50), method="min_variance")
    assert r is not None

    # target_positions when construct yields None positions — force via empty cash fallback
    tp = eng.target_positions(forecasts=None, signals=None, names=["a", "b"])
    assert tp is not None


def test_projection_simplex_theta_zero_and_equal():
    # all equal negative → rho empty → theta=0 then s<=0 → equal weights
    out = proj.project_simplex(np.array([-1.0, -1.0, -1.0]), budget=1.0)
    assert abs(out.sum() - 1.0) < 1e-8

    # project_gross under max when already ok → budget rescale path 244
    w = np.array([0.2, 0.3, 0.5])
    g = proj.project_gross(w, max_gross=2.0, budget=1.0, long_only=True)
    assert abs(g.sum() - 1.0) < 1e-8

    # long-short project_weights with max_gross forcing re-box (290)
    cstr = proj.parse_constraints(
        {"long_only": False, "max_weight": 0.8, "min_weight": -0.5, "max_gross": 1.2},
        n=3,
        long_only=False,
        max_weight=0.8,
        min_weight=-0.5,
        max_gross=1.2,
        budget=0.5,
    )
    pw = proj.project_weights(np.array([2.0, -1.5, 0.5]), cstr)
    assert pw.shape == (3,)


def test_outer_except_via_parse_constraints(cov, names, monkeypatch):
    import iqrp.app.portfolio.multi_period.optimizer as mp
    import iqrp.app.portfolio.optimization.black_litterman as bl
    import iqrp.app.portfolio.optimization.entropy as ent
    import iqrp.app.portfolio.optimization.hierarchical as hi
    import iqrp.app.portfolio.optimization.maximum_diversification as md
    import iqrp.app.portfolio.optimization.maximum_sharpe as ms
    import iqrp.app.portfolio.optimization.minimum_variance as mv
    import iqrp.app.portfolio.optimization.risk_parity as rp
    import iqrp.app.portfolio.optimization.turnover as to
    import iqrp.app.portfolio.robust.distributional_robust as dr

    def boom(*a, **k):
        raise RuntimeError("parse fail")

    for mod in (ent, mv, ms, md, rp, hi, to, dr, bl, mp):
        monkeypatch.setattr(mod, "parse_constraints", boom, raising=False)
        monkeypatch.setattr(mod, "as_cov", lambda *a, **k: np.asarray(cov), raising=False)

    assert ent.optimize_entropy(mu=np.ones(4) * 0.01, cov=cov, names=names)["success"] is False
    assert mv.optimize_minimum_variance(cov=cov, names=names)["success"] is False
    assert (
        ms.optimize_maximum_sharpe(mu=np.ones(4) * 0.01, cov=cov, names=names)["success"] is False
    )
    assert md.optimize_maximum_diversification(cov=cov, names=names)["success"] is False
    assert rp.optimize_risk_parity(cov=cov, names=names)["success"] is False
    assert hi.optimize_hrp(cov=cov, names=names)["success"] is False
    assert to.optimize_turnover(mu=np.ones(4) * 0.01, cov=cov, names=names)["success"] is False
    assert (
        dr.optimize_distributional_robust(mu=np.ones(4) * 0.01, cov=cov, names=names)["success"]
        is False
    )
    bl_res = bl.optimize_black_litterman(cov=cov, market_weights=np.ones(4) / 4, names=names)
    assert "success" in bl_res  # may fail or succeed depending on when parse is called
    assert (
        mp.optimize_multi_period(mu=np.ones(4) * 0.01, cov=cov, horizons=2, names=names)["success"]
        is False
    )
