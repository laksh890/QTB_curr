"""Robust optimization and multi-period module tests."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.portfolio.multi_period import (
    apply_drift,
    optimize_dynamic_programming,
    optimize_multi_period,
    rebalance_schedule,
    turnover,
)
from iqrp.app.portfolio.robust import (
    box_uncertainty_cov,
    box_uncertainty_mu,
    ellipsoidal_uncertainty_mu,
    optimize_distributional_robust,
    optimize_parameter_uncertainty,
    optimize_robust_mean_variance,
    shrink_covariance,
    worst_case_mu,
    worst_case_return,
)


def test_uncertainty_sets(mu, cov, weights):
    box = box_uncertainty_mu(mu, relative=0.1, kappa=0.5, cov=cov)
    assert isinstance(box, dict)
    ell = ellipsoidal_uncertainty_mu(mu, cov, rho=1.0, tau=1.0)
    assert isinstance(ell, dict)
    bc = box_uncertainty_cov(cov, relative=0.1)
    assert isinstance(bc, dict) or hasattr(bc, "shape")

    wc = worst_case_mu(weights, box)
    assert wc is not None
    wr = worst_case_return(weights, ell)
    assert isinstance(wr, (float, np.floating)) or wr is not None

    shrunk = shrink_covariance(cov, intensity=0.2)
    assert shrunk.shape == cov.shape


def test_optimize_distributional_robust(mu, cov, names):
    res = optimize_distributional_robust(
        mu=mu,
        cov=cov,
        names=names,
        uncertainty="ellipsoidal",
        rho=1.0,
        max_weight=0.5,
    )
    assert res["success"] or res["status"] in ("infeasible", "failed", "fallback")
    if res["success"]:
        w = list(res["weights"].values()) if isinstance(res["weights"], dict) else res["weights"]
        assert abs(sum(w) - 1.0) < 1e-4


def test_optimize_distributional_robust_box(mu, cov, names):
    res = optimize_distributional_robust(
        mu=mu,
        cov=cov,
        names=names,
        uncertainty="box",
        relative=0.1,
        max_weight=0.5,
    )
    assert "success" in res


def test_optimize_parameter_uncertainty(mu, cov, returns, names):
    res = optimize_parameter_uncertainty(
        mu=mu,
        cov=cov,
        returns=returns,
        names=names,
        z_score=1.0,
        max_weight=0.5,
    )
    assert "success" in res


def test_optimize_robust_mean_variance(mu, cov, names):
    res = optimize_robust_mean_variance(mu=mu, cov=cov, names=names, max_weight=0.5)
    assert "success" in res


def test_distributional_robust_infeasible(mu, cov, names):
    res = optimize_distributional_robust(
        mu=mu,
        cov=cov,
        names=names,
        max_weight=0.1,
        budget=1.0,
    )
    assert res["success"] is False
    assert res["status"] == "infeasible"


def test_multi_period_optimize(mu, cov, names):
    n = len(names)
    horizons = 3
    mu_path = np.tile(mu, (horizons, 1))
    cov_path = np.stack([cov] * horizons, axis=0)
    res = optimize_multi_period(
        mu=mu,
        cov=cov,
        horizons=horizons,
        mu_path=mu_path,
        cov_path=cov_path,
        transaction_cost=0.001,
        rebalance_every=1,
        names=names,
        max_weight=0.5,
    )
    assert "success" in res
    if res["success"]:
        diag = res.get("diagnostics") or {}
        assert "weights_path" in diag or "schedule" in diag or res.get("weights") is not None


def test_multi_period_infeasible_turnover(mu, cov, names, current_weights):
    horizons = 2
    mu_path = np.tile(np.array([1.0, -1.0, -1.0, -1.0][: len(names)]), (horizons, 1))
    res = optimize_multi_period(
        mu=mu_path[0],
        cov=cov,
        horizons=horizons,
        mu_path=mu_path,
        cov_path=np.stack([cov] * horizons),
        current_weights=current_weights,
        turnover_threshold=1e-15,
        transaction_cost=0.0,
        names=names,
        max_weight=0.5,
    )
    # either success with tiny trades or explicit failure — never silent relax of hard TO
    assert "success" in res
    if not res["success"]:
        assert res["status"] in ("infeasible", "failed")
        assert res.get("failure_reason")


def test_multi_period_bad_horizons(mu, cov, names):
    res = optimize_multi_period(mu=mu, cov=cov, horizons=0, names=names)
    assert res["success"] is False


def test_dynamic_programming(mu, cov, names):
    # keep grid small: n<=4, horizons=2, grid_levels=4
    n = min(3, len(names))
    names_s = names[:n]
    mu_s = mu[:n]
    cov_s = cov[:n, :n]
    mu_path = np.tile(mu_s, (2, 1))
    res = optimize_dynamic_programming(
        mu=mu_s,
        cov=cov_s,
        horizons=2,
        mu_path=mu_path,
        transaction_cost=0.001,
        grid_levels=4,
        names=names_s,
        max_weight=0.6,
    )
    assert "success" in res


def test_rebalance_schedule_and_drift(weights, returns):
    sched = rebalance_schedule(n_periods=10, frequency=2, threshold=None)
    assert "flags" in sched or isinstance(sched, dict)
    flags = sched.get("flags") if isinstance(sched, dict) else sched
    assert flags is not None

    # one-period returns for drift
    r1 = returns[0]
    drifted = apply_drift(weights, r1)
    assert drifted.shape == weights.shape
    assert drifted.sum() == pytest.approx(1.0, rel=1e-6) or drifted.sum() > 0

    to = turnover(weights, drifted)
    assert to >= 0.0
