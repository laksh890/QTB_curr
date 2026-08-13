"""Maximum diversification ratio portfolio."""

from __future__ import annotations

from typing import Any

import numpy as np

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
    portfolio_variance,
    project_weights,
    projected_gradient,
    scipy_available,
)


def optimize_maximum_diversification(
    mu: Any = None,
    cov: Any = None,
    *,
    current_weights: Any = None,
    constraints: Any = None,
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_aversion: float = 1.0,
    min_weight: float | None = None,
    max_gross: float | None = None,
    budget: float = 1.0,
    ridge: float = 1e-8,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Maximize diversification ratio (w'σ) / sqrt(w'Σw)."""
    name = "maximum_diversification"
    method = "projected_gradient"
    _ = mu
    _ = risk_aversion
    try:
        if cov is None:
            raise ValueError("cov is required")
        c = as_cov(cov)
        n = c.shape[0]
        c = c + float(ridge) * np.eye(n)
        vols = np.sqrt(np.maximum(np.diag(c), 1e-18))

        cstr = parse_constraints(
            constraints,
            n,
            long_only=long_only,
            max_weight=max_weight,
            min_weight=min_weight,
            max_gross=max_gross,
            budget=budget,
        )
        if names is None:
            names = cstr.get("names")
        ok, reason, conflicts = check_feasibility(cstr)
        if not ok:
            return infeasible_result(name, n, method=method, reason=reason or "infeasible", conflicts=conflicts, names=names)

        def project(w: np.ndarray) -> np.ndarray:
            return project_weights(w, cstr)

        # Inverse-vol seed is a strong diversification prior
        seed = (1.0 / vols)
        seed = seed / float(np.sum(seed)) * cstr["budget"]
        x0 = project(as_vector(current_weights, n) if current_weights is not None else seed)

        def neg_dr(w: np.ndarray) -> float:
            num = float(w @ vols)
            den = float(np.sqrt(max(portfolio_variance(w, c), 1e-18)))
            return -num / den

        def grad_ndr(w: np.ndarray) -> np.ndarray:
            num = float(w @ vols)
            var = max(portfolio_variance(w, c), 1e-18)
            vol = float(np.sqrt(var))
            d_num = vols
            d_vol = (c @ w) / vol
            d_dr = (d_num * vol - num * d_vol) / (vol * vol)
            return -d_dr

        used = "numpy_pgd"
        if scipy_available():
            bounds = [(cstr["lb"], cstr["ub"])] * n
            cons = {"type": "eq", "fun": lambda ww: float(np.sum(ww) - cstr["budget"])}
            try:
                res = minimize_scipy(neg_dr, x0, jac=grad_ndr, bounds=bounds, constraints=[cons], method="SLSQP")
                if bool(res.success):
                    w = project(np.asarray(res.x, dtype=np.float64))
                    fval = float(neg_dr(w))
                    used = "scipy_slsqp"
                    success_opt = True
                    iters = int(getattr(res, "nit", 0) or 0)
                else:
                    w, fval, success_opt, iters = projected_gradient(neg_dr, grad_ndr, x0, project, lr=0.05)
            except Exception:
                w, fval, success_opt, iters = projected_gradient(neg_dr, grad_ndr, x0, project, lr=0.05)
        else:
            w, fval, success_opt, iters = projected_gradient(neg_dr, grad_ndr, x0, project, lr=0.05)

        if float(np.min(w)) < cstr["lb"] - 1e-8 or float(np.max(w)) > cstr["ub"] + 1e-8:
            return infeasible_result(name, n, method=used, reason="box violation", conflicts=["box"], names=names)
        if abs(float(np.sum(w)) - cstr["budget"]) > 1e-6:
            return infeasible_result(name, n, method=used, reason="budget violation", conflicts=["budget"], names=names)

        dr = -fval
        return make_result(
            name,
            format_weights(w, names),
            success=True,
            status="optimal" if success_opt else "fallback",
            method=used,
            diagnostics={
                "n_assets": n,
                "diversification_ratio": dr,
                "weighted_vol": float(w @ vols),
                "portfolio_vol": float(np.sqrt(max(portfolio_variance(w, c), 0.0))),
                "iterations": iters,
            },
            objective_value=dr,
        )
    except Exception as exc:
        n = 0
        try:
            n = int(np.asarray(cov).shape[0]) if cov is not None else 0
        except Exception:
            n = 0
        return failed_result(name, n, method=method, reason=str(exc))
