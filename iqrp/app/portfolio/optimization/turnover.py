"""Turnover-penalized mean-variance optimization."""

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


def optimize_turnover(
    mu: Any = None,
    cov: Any = None,
    *,
    current_weights: Any = None,
    constraints: Any = None,
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_aversion: float = 1.0,
    turnover_penalty: float = 0.01,
    max_turnover: float | None = None,
    min_weight: float | None = None,
    max_gross: float | None = None,
    budget: float = 1.0,
    ridge: float = 1e-8,
    names: list[str] | None = None,
    mu_clip: float = 0.5,
) -> dict[str, Any]:
    """
    Maximize w'μ - (λ/2) w'Σw - τ ||w - w0||_1.

    If ``max_turnover`` is set it is a hard constraint; violation → infeasible
    (never silently relaxed).
    """
    name = "turnover"
    method = "turnover_penalized_mv"
    try:
        if cov is None:
            raise ValueError("cov is required")
        c = as_cov(cov)
        n = c.shape[0]
        c = c + float(ridge) * np.eye(n)
        m = np.zeros(n) if mu is None else stabilize_mu(as_vector(mu, n), clip=mu_clip)

        cstr = parse_constraints(
            constraints,
            n,
            long_only=long_only,
            max_weight=max_weight,
            min_weight=min_weight,
            max_gross=max_gross,
            budget=budget,
        )
        # allow max_turnover via constraints dict
        mt = max_turnover
        if isinstance(constraints, dict) and constraints.get("max_turnover") is not None:
            mt = float(constraints["max_turnover"])
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

        if current_weights is not None:
            raw = as_vector(current_weights, n)
            s = float(np.sum(raw))
            w0 = raw / s * cstr["budget"] if abs(s) > 1e-14 else equal_weights(n, cstr["budget"])
        else:
            w0 = equal_weights(n, cstr["budget"])

        lam = max(float(risk_aversion), 1e-8)
        tau = max(float(turnover_penalty), 0.0)

        def project(w: np.ndarray) -> np.ndarray:
            return project_weights(w, cstr)

        x0 = project(w0.copy())

        def turnover(w: np.ndarray) -> float:
            return 0.5 * float(np.sum(np.abs(w - w0)))

        def obj(w: np.ndarray) -> float:
            return (
                0.5 * lam * portfolio_variance(w, c)
                - portfolio_return(w, m)
                + tau * float(np.sum(np.abs(w - w0)))
            )

        def grad(w: np.ndarray) -> np.ndarray:
            # subgradient of L1
            return lam * (c @ w) - m + tau * np.sign(w - w0)

        used = "numpy_pgd"
        if scipy_available():
            bounds = [(cstr["lb"], cstr["ub"])] * n
            cons = [{"type": "eq", "fun": lambda ww: float(np.sum(ww) - cstr["budget"])}]
            if mt is not None:
                cons.append(
                    {
                        "type": "ineq",
                        "fun": lambda ww, _w0=w0, _mt=float(mt): float(_mt)
                        - 0.5 * float(np.sum(np.abs(ww - _w0))),
                    }
                )
            try:
                res = minimize_scipy(
                    obj, x0, jac=grad, bounds=bounds, constraints=cons, method="SLSQP"
                )
                if bool(res.success):
                    w = project(np.asarray(res.x, dtype=np.float64))
                    fval = float(obj(w))
                    used = "scipy_slsqp"
                    success_opt = True
                    iters = int(getattr(res, "nit", 0) or 0)
                else:
                    w, fval, success_opt, iters = projected_gradient(
                        obj, grad, x0, project, lr=0.05
                    )
                    used = "numpy_pgd"
            except Exception:
                w, fval, success_opt, iters = projected_gradient(obj, grad, x0, project, lr=0.05)
                used = "numpy_pgd"
        else:
            w, fval, success_opt, iters = projected_gradient(obj, grad, x0, project, lr=0.05)

        to = turnover(w)
        if mt is not None and to > float(mt) + 1e-6:
            # Hard constraint: attempt projection toward w0 along L1 path
            # Binary search blend with w0
            lo, hi = 0.0, 1.0
            w_feas = w0.copy()
            found = False
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                cand = project((1.0 - mid) * w0 + mid * w)
                t_c = turnover(cand)
                if t_c <= float(mt) + 1e-8:
                    w_feas = cand
                    found = True
                    lo = mid
                else:
                    hi = mid
            if not found or turnover(w_feas) > float(mt) + 1e-6:
                return infeasible_result(
                    name,
                    n,
                    method=used,
                    reason=f"max_turnover={mt} cannot be satisfied from current_weights",
                    conflicts=["max_turnover"],
                    names=names,
                )
            w = w_feas
            to = turnover(w)
            fval = float(obj(w))
            used = used + "_turnover_projection"

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
                "turnover": to,
                "turnover_penalty": tau,
                "max_turnover": mt,
                "expected_return": portfolio_return(w, m),
                "variance": portfolio_variance(w, c),
                "iterations": iters,
            },
            objective_value=-fval,
        )
    except Exception as exc:
        n = 0
        try:
            n = int(np.asarray(cov).shape[0]) if cov is not None else 0
        except Exception:
            n = 0
        return failed_result(name, n, method=method, reason=str(exc))
