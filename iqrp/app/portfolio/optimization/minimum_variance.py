"""Global minimum-variance portfolio optimization."""

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


def optimize_minimum_variance(
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
    """
    Minimize w'Σw subject to budget and box constraints.

    Unconstrained closed form is projected onto the feasible set when long_only=False
    or when box bounds apply.
    """
    name = "minimum_variance"
    method = "analytic_projection"
    _ = mu  # unused; signature compatibility
    _ = risk_aversion
    try:
        if cov is None:
            raise ValueError("cov is required")
        c = as_cov(cov)
        n = c.shape[0]
        c = c + float(ridge) * np.eye(n)

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
            return infeasible_result(
                name,
                n,
                method=method,
                reason=reason or "infeasible",
                conflicts=conflicts,
                names=names,
            )

        ones = np.ones(n)
        try:
            inv_ones = np.linalg.solve(c, ones)
            w_unc = inv_ones / float(np.sum(inv_ones))
            w_unc = w_unc * cstr["budget"]
        except np.linalg.LinAlgError:
            w_unc = equal_weights(n, cstr["budget"])

        def project(w: np.ndarray) -> np.ndarray:
            return project_weights(w, cstr)

        x0 = project(as_vector(current_weights, n) if current_weights is not None else w_unc)

        def obj(w: np.ndarray) -> float:
            return portfolio_variance(w, c)

        def grad(w: np.ndarray) -> np.ndarray:
            return 2.0 * (c @ w)

        used = "analytic_projection"
        w = project(w_unc)
        # If analytic already feasible under box, keep it; else refine
        needs_opt = (
            float(np.min(w_unc)) < cstr["lb"] - 1e-10
            or float(np.max(w_unc)) > cstr["ub"] + 1e-10
            or (
                cstr.get("max_gross") is not None
                and float(np.sum(np.abs(w_unc))) > float(cstr["max_gross"]) + 1e-10
            )
        )
        fval = float(obj(w))
        success_opt = True
        iters = 0
        if needs_opt:
            if scipy_available():
                bounds = [(cstr["lb"], cstr["ub"])] * n
                cons = {"type": "eq", "fun": lambda ww: float(np.sum(ww) - cstr["budget"])}
                try:
                    res = minimize_scipy(
                        obj, x0, jac=grad, bounds=bounds, constraints=[cons], method="SLSQP"
                    )
                    if bool(res.success):
                        w = project(np.asarray(res.x, dtype=np.float64))
                        fval = float(obj(w))
                        used = "scipy_slsqp"
                        iters = int(getattr(res, "nit", 0) or 0)
                    else:
                        w, fval, success_opt, iters = projected_gradient(
                            obj, grad, x0, project, lr=0.1
                        )
                        used = "numpy_pgd"
                except Exception:
                    w, fval, success_opt, iters = projected_gradient(obj, grad, x0, project, lr=0.1)
                    used = "numpy_pgd"
            else:
                w, fval, success_opt, iters = projected_gradient(obj, grad, x0, project, lr=0.1)
                used = "numpy_pgd"

        # hard constraint verification
        if float(np.min(w)) < cstr["lb"] - 1e-8 or float(np.max(w)) > cstr["ub"] + 1e-8:
            return infeasible_result(
                name, n, method=used, reason="box violation", conflicts=["box"], names=names
            )
        if abs(float(np.sum(w)) - cstr["budget"]) > 1e-6:
            return infeasible_result(
                name, n, method=used, reason="budget violation", conflicts=["budget"], names=names
            )

        return make_result(
            name,
            format_weights(w, names),
            success=True,
            status="optimal" if success_opt else "fallback",
            method=used,
            diagnostics={
                "n_assets": n,
                "variance": fval,
                "volatility": float(np.sqrt(max(fval, 0.0))),
                "analytic_used": not needs_opt,
                "iterations": iters,
                "long_only": bool(cstr["long_only"]),
            },
            objective_value=fval,
        )
    except Exception as exc:
        n = 0
        try:
            n = int(np.asarray(cov).shape[0]) if cov is not None else 0
        except Exception:
            n = 0
        return failed_result(name, n, method=method, reason=str(exc))
