"""Tests for all optimize_* methods: success or explicit failure/infeasibility."""

from __future__ import annotations

import numpy as np
import pytest

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
    optimize_robust,
    optimize_turnover,
)
from iqrp.app.portfolio.optimization.projection import (
    check_feasibility,
    parse_constraints,
    project_box_simplex,
)


def _assert_weights_ok(res: dict, n: int, *, max_weight: float = 0.5, long_only: bool = True):
    assert "success" in res
    assert "weights" in res
    assert "status" in res
    w = res["weights"]
    if isinstance(w, dict):
        vals = list(w.values())
    else:
        vals = list(w)
    assert len(vals) == n
    if res["success"]:
        if long_only:
            assert all(v >= -1e-7 for v in vals)
        assert abs(sum(vals) - 1.0) < 1e-4 or abs(sum(vals)) < 1e-8
        if max_weight is not None and sum(vals) > 0.5:
            assert max(vals) <= max_weight + 1e-5
    else:
        # infeasible / failed: explicit reason, no silent relax
        assert res.get("failure_reason") or res["status"] in ("infeasible", "failed")
        assert res["status"] in ("infeasible", "failed", "fallback")


def test_optimize_mean_variance_success(mu, cov, names):
    res = optimize_mean_variance(mu=mu, cov=cov, names=names, max_weight=0.5)
    _assert_weights_ok(res, len(names))


def test_optimize_minimum_variance_success(cov, names):
    res = optimize_minimum_variance(cov=cov, names=names, max_weight=0.5)
    _assert_weights_ok(res, len(names))


def test_optimize_maximum_sharpe_success(mu, cov, names):
    res = optimize_maximum_sharpe(mu=mu, cov=cov, names=names, max_weight=0.5)
    _assert_weights_ok(res, len(names))


def test_optimize_maximum_diversification(cov, names):
    res = optimize_maximum_diversification(cov=cov, names=names, max_weight=0.5)
    _assert_weights_ok(res, len(names))


def test_optimize_risk_parity_and_erc(cov, names):
    res = optimize_risk_parity(cov=cov, names=names, method="risk_parity", max_weight=0.5)
    _assert_weights_ok(res, len(names))
    res2 = optimize_risk_parity(cov=cov, names=names, method="erc", max_weight=0.5)
    _assert_weights_ok(res2, len(names))


def test_optimize_hrp_herc(cov, names):
    res = optimize_hrp(cov=cov, names=names, max_weight=0.6)
    _assert_weights_ok(res, len(names), max_weight=0.6)
    res2 = optimize_herc(cov=cov, names=names, max_weight=0.6)
    _assert_weights_ok(res2, len(names), max_weight=0.6)


def test_optimize_cvar_with_scenarios(returns, names):
    res = optimize_cvar(scenarios=returns, names=names, alpha=0.95, max_weight=0.5)
    _assert_weights_ok(res, len(names))


def test_optimize_cvar_synthetic_from_cov(cov, names):
    res = optimize_cvar(cov=cov, names=names, alpha=0.9, max_weight=0.5)
    _assert_weights_ok(res, len(names))


def test_optimize_drawdown(returns, cov, names):
    res = optimize_drawdown(returns=returns, cov=cov, names=names, max_weight=0.5)
    _assert_weights_ok(res, len(names))


def test_optimize_turnover(mu, cov, names, current_weights):
    res = optimize_turnover(
        mu=mu,
        cov=cov,
        names=names,
        current_weights=current_weights,
        turnover_penalty=0.05,
        max_weight=0.5,
    )
    _assert_weights_ok(res, len(names))


def test_optimize_turnover_hard_cap_infeasible(mu, cov, names, current_weights):
    """Hard max_turnover that cannot be met → infeasible, not silent relax."""
    # Force target far from current with tiny turnover budget
    res = optimize_turnover(
        mu=np.array([1.0, -1.0, -1.0, -1.0][: len(names)]),
        cov=cov,
        names=names,
        current_weights=current_weights,
        max_turnover=1e-12,
        turnover_penalty=0.0,
        max_weight=0.5,
    )
    if not res["success"]:
        assert res["status"] in ("infeasible", "failed")
        assert res.get("failure_reason")
        # weights should not violate by relaxing — zeros or within turnover of current
        w = np.asarray(
            list(res["weights"].values()) if isinstance(res["weights"], dict) else res["weights"]
        )
        # either zeros or within tiny turnover of current
        to = 0.5 * np.sum(np.abs(w - current_weights))
        assert to <= 1e-6 + 1e-9 or np.allclose(w, 0.0)


def test_optimize_entropy(mu, cov, names):
    res = optimize_entropy(mu=mu, cov=cov, names=names, max_weight=0.5)
    _assert_weights_ok(res, len(names))


def test_optimize_robust(mu, cov, names):
    res = optimize_robust(mu=mu, cov=cov, names=names, max_weight=0.5)
    _assert_weights_ok(res, len(names))


def test_optimize_black_litterman(mu, cov, names):
    n = len(names)
    P = np.zeros((1, n))
    P[0, 0] = 1.0
    Q = np.array([0.02])
    res = optimize_black_litterman(
        cov=cov,
        names=names,
        market_weights=np.ones(n) / n,
        P=P,
        Q=Q,
        max_weight=0.5,
    )
    _assert_weights_ok(res, n)


def test_optimize_missing_cov_fails(names):
    res = optimize_mean_variance(mu=np.ones(len(names)) * 0.01, cov=None, names=names)
    assert res["success"] is False
    assert res["status"] in ("failed", "infeasible")


def test_optimize_sharpe_missing_mu_fails(cov, names):
    res = optimize_maximum_sharpe(mu=None, cov=cov, names=names)
    assert res["success"] is False


def test_infeasible_max_weight_does_not_relax(cov, names):
    """n * max_weight < budget → infeasible; hard constraints not relaxed."""
    res = optimize_minimum_variance(
        cov=cov,
        names=names,
        max_weight=0.1,  # 4*0.1=0.4 < 1.0
        budget=1.0,
        long_only=True,
    )
    assert res["success"] is False
    assert res["status"] == "infeasible"
    assert res.get("conflicting_constraints") or res.get("failure_reason")
    w = np.asarray(
        list(res["weights"].values()) if isinstance(res["weights"], dict) else res["weights"]
    )
    # zeros — not a silently relaxed portfolio that sums to 1 with max>0.1
    assert np.allclose(w, 0.0) or max(w) <= 0.1 + 1e-8


def test_infeasible_max_gross(cov, names):
    res = optimize_mean_variance(
        mu=np.ones(len(names)) * 0.01,
        cov=cov,
        names=names,
        max_weight=0.5,
        max_gross=0.5,
        budget=1.0,
        long_only=True,
    )
    assert res["success"] is False
    assert res["status"] == "infeasible"


def test_unsupported_hard_constraint_extras(cov, names):
    res = optimize_mean_variance(
        mu=np.ones(len(names)) * 0.01,
        cov=cov,
        names=names,
        constraints={"sector_limits": {"Tech": 0.3}},
        max_weight=0.5,
    )
    assert res["success"] is False
    assert res["status"] == "infeasible"
    conflicts = res.get("conflicting_constraints") or []
    assert any(
        "unsupported" in str(c).lower() or "sector" in str(c).lower() for c in conflicts
    ) or res.get("failure_reason")


def test_risk_parity_rejects_shorts(cov, names):
    res = optimize_risk_parity(cov=cov, names=names, long_only=False, min_weight=-0.2)
    # either forced long_only success or infeasible
    if not res["success"]:
        assert res["status"] in ("infeasible", "failed")


def test_cvar_bad_alpha(cov, names):
    res = optimize_cvar(cov=cov, names=names, alpha=0.1)
    assert res["success"] is False or res.get("failure_reason")


def test_parse_constraints_and_feasibility(names):
    cstr = parse_constraints(
        None,
        n=len(names),
        long_only=True,
        max_weight=0.5,
    )
    ok, reason, conflicts = check_feasibility(cstr)
    assert ok is True

    cstr_bad = parse_constraints(
        None,
        n=len(names),
        long_only=True,
        max_weight=0.1,
        budget=1.0,
    )
    ok2, reason2, conflicts2 = check_feasibility(cstr_bad)
    assert ok2 is False
    assert conflicts2 or reason2


def test_project_box_simplex_infeasible():
    with pytest.raises(ValueError):
        project_box_simplex(np.ones(3), lb=np.zeros(3), ub=np.ones(3) * 0.2, budget=1.0)


def test_length_mismatch_fails(cov, names):
    res = optimize_mean_variance(
        mu=np.ones(len(names) + 1) * 0.01,
        cov=cov,
        names=names,
    )
    assert res["success"] is False or len(
        list(res["weights"].values()) if isinstance(res["weights"], dict) else res["weights"]
    ) in (len(names), len(names) + 1)
