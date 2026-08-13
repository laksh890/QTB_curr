"""Mean-variance (Markowitz) portfolio optimization."""

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
    portfolio_return,
    portfolio_variance,
    project_weights,
    projected_gradient,
    scipy_available,
    stabilize_mu,
)


def optimize_mean_variance(
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
    mu_clip: float = 0.5,
) -> dict[str, Any]:
    """
    Maximize w'μ - (λ/2) w'Σw subject to box + budget constraints.

    Hard constraints are never silently relaxed; infeasible inputs return success=False.
    """
    name = "mean_variance"
    method = "projected_gradient"
    try:
        if cov is None:
            raise ValueError("cov is required")
        c = as_cov(cov)
        n = c.shape[0]
        c = c + float(ridge) * np.eye(n)
        if mu is None:
            m = np.zeros(n)
        else:
            m = stabilize_mu(as_vector(mu, n, name="mu"), clip=mu_clip)

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

        lam = max(float(risk_aversion), 1e-8)

        def obj(w: np.ndarray) -> float:
            return 0.5 * lam * portfolio_variance(w, c) - portfolio_return(w, m)

        def grad(w: np.ndarray) -> np.ndarray:
            return lam * (c @ w) - m

        def project(w: np.ndarray) -> np.ndarray:
            return project_weights(w, cstr)

        if current_weights is not None:
            x0 = project(as_vector(current_weights, n, name="current_weights"))
        else:
            # analytic unconstrained direction then project
            try:
                inv = np.linalg.solve(c, m)
                x0 = project(inv / lam)
            except np.linalg.LinAlgError:
                x0 = project(equal_weights(n, cstr["budget"]))

        used = "numpy_pgd"
        w = x0
        fval = float(obj(w))
        success_opt = True
        iters = 0

        if scipy_available():
            bounds = [(cstr["lb"], cstr["ub"])] * n
            cons = {"type": "eq", "fun": lambda w: float(np.sum(w) - cstr["budget"])}
            try:
                res = minimize_scipy(obj, x0, jac=grad, bounds=bounds, constraints=[cons], method="SLSQP")
                if bool(res.success):
                    w = project(np.asarray(res.x, dtype=np.float64))
                    fval = float(obj(w))
                    used = "scipy_slsqp"
                    success_opt = True
                    iters = int(getattr(res, "nit", 0) or 0)
                else:
                    w, fval, success_opt, iters = projected_gradient(obj, grad, x0, project)
                    used = "numpy_pgd_fallback"
            except Exception:
                w, fval, success_opt, iters = projected_gradient(obj, grad, x0, project)
                used = "numpy_pgd_fallback"
        else:
            w, fval, success_opt, iters = projected_gradient(obj, grad, x0, project)

        # Verify hard constraints; do not relax
        if float(np.min(w)) < cstr["lb"] - 1e-8 or float(np.max(w)) > cstr["ub"] + 1e-8:
            return infeasible_result(
                name,
                n,
                method=used,
                reason="optimizer violated box constraints",
                conflicts=["box"],
                names=names,
            )
        if abs(float(np.sum(w)) - cstr["budget"]) > 1e-6:
            return infeasible_result(
                name,
                n,
                method=used,
                reason="optimizer violated budget constraint",
                conflicts=["budget"],
                names=names,
            )
        gross = cstr.get("max_gross")
        if gross is not None and float(np.sum(np.abs(w))) > float(gross) + 1e-6:
            return infeasible_result(
                name,
                n,
                method=used,
                reason="optimizer violated max_gross",
                conflicts=["max_gross"],
                names=names,
            )

        status = "optimal" if success_opt else "fallback"
        return make_result(
            name,
            format_weights(w, names),
            success=True,
            status=status,
            method=used,
            diagnostics={
                "n_assets": n,
                "risk_aversion": lam,
                "expected_return": portfolio_return(w, m),
                "variance": portfolio_variance(w, c),
                "volatility": float(np.sqrt(max(portfolio_variance(w, c), 0.0))),
                "iterations": iters,
                "mu_stabilized": True,
            },
            objective_value=-fval,  # report utility
        )
    except Exception as exc:
        n = 0
        try:
            n = int(np.asarray(cov).shape[0]) if cov is not None else 0
        except Exception:
            n = 0
        return failed_result(name, n, method=method, reason=str(exc))
