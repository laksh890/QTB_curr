"""Distributionally robust mean-variance style optimization."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.portfolio.optimization.mean_variance import optimize_mean_variance
from iqrp.app.portfolio.optimization.projection import (
    as_cov,
    as_vector,
    check_feasibility,
    equal_weights,
    failed_result,
    format_weights,
    infeasible_result,
    make_result,
    parse_constraints,
    portfolio_variance,
    project_weights,
    projected_gradient,
    scipy_available,
    minimize_scipy,
    stabilize_mu,
)
from iqrp.app.portfolio.robust.uncertainty_sets import (
    box_uncertainty_mu,
    ellipsoidal_uncertainty_mu,
    worst_case_mu,
    worst_case_return,
)


def optimize_distributional_robust(
    mu: Any = None,
    cov: Any = None,
    *,
    current_weights: Any = None,
    constraints: Any = None,
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_aversion: float = 1.0,
    uncertainty: str | dict[str, Any] = "ellipsoidal",
    rho: float = 1.0,
    tau: float = 0.05,
    relative: float = 0.1,
    kappa: float = 0.5,
    min_weight: float | None = None,
    max_gross: float | None = None,
    budget: float = 1.0,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Worst-case expected return in an uncertainty set, then mean-variance.

    For ellipsoidal sets the objective becomes:
        max_w  w'μ - rho * sqrt(w'(tau Σ)w) - (λ/2) w'Σw
    """
    name = "distributional_robust"
    method = "worst_case_mv"
    try:
        if cov is None:
            raise ValueError("cov is required")
        c = as_cov(cov)
        n = c.shape[0]
        m = np.zeros(n) if mu is None else stabilize_mu(as_vector(mu, n))

        if isinstance(uncertainty, dict):
            uset = uncertainty
        else:
            kind = str(uncertainty).lower()
            if kind in {"box", "interval"}:
                uset = box_uncertainty_mu(m, relative=relative, kappa=kappa, cov=c)
            else:
                uset = ellipsoidal_uncertainty_mu(m, c, rho=rho, tau=tau)

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

        # Seed with worst-case mu plug-in MV
        w_seed = equal_weights(n, cstr["budget"])
        if current_weights is not None:
            w_seed = as_vector(current_weights, n)
        mu_wc = worst_case_mu(w_seed, uset)
        seed = optimize_mean_variance(
            mu=mu_wc,
            cov=c,
            current_weights=w_seed,
            constraints={
                "long_only": cstr["long_only"],
                "max_weight": cstr["ub"],
                "min_weight": cstr["lb"],
                "max_gross": cstr.get("max_gross"),
                "budget": cstr["budget"],
            },
            long_only=cstr["long_only"],
            max_weight=cstr["ub"],
            risk_aversion=risk_aversion,
            min_weight=cstr["lb"],
            max_gross=cstr.get("max_gross"),
            budget=cstr["budget"],
        )
        x0 = project_weights(as_vector(seed["weights"], n) if seed.get("success") else w_seed, cstr)

        lam = max(float(risk_aversion), 1e-8)

        def project(w: np.ndarray) -> np.ndarray:
            return project_weights(w, cstr)

        def obj(w: np.ndarray) -> float:
            # minimize negative worst-case utility
            wc_ret = worst_case_return(w, uset)
            return 0.5 * lam * portfolio_variance(w, c) - wc_ret

        def grad(w: np.ndarray) -> np.ndarray:
            # use worst-case mu linearization at current w
            mu_lin = worst_case_mu(w, uset)
            return lam * (c @ w) - mu_lin

        used = "numpy_pgd"
        if scipy_available():
            bounds = [(cstr["lb"], cstr["ub"])] * n
            cons = {"type": "eq", "fun": lambda ww: float(np.sum(ww) - cstr["budget"])}
            try:
                res = minimize_scipy(obj, x0, jac=grad, bounds=bounds, constraints=[cons], method="SLSQP")
                if bool(res.success):
                    w = project(np.asarray(res.x, dtype=np.float64))
                    fval = float(obj(w))
                    used = "scipy_slsqp"
                    success_opt = True
                    iters = int(getattr(res, "nit", 0) or 0)
                else:
                    w, fval, success_opt, iters = projected_gradient(obj, grad, x0, project, lr=0.05)
            except Exception:
                w, fval, success_opt, iters = projected_gradient(obj, grad, x0, project, lr=0.05)
        else:
            w, fval, success_opt, iters = projected_gradient(obj, grad, x0, project, lr=0.05)

        if float(np.min(w)) < cstr["lb"] - 1e-8 or float(np.max(w)) > cstr["ub"] + 1e-8:
            return infeasible_result(name, n, method=used, reason="box violation", conflicts=["box"], names=names)
        if abs(float(np.sum(w)) - cstr["budget"]) > 1e-6:
            return infeasible_result(name, n, method=used, reason="budget violation", conflicts=["budget"], names=names)

        return make_result(
            name,
            format_weights(w, names),
            success=True,
            status="optimal" if success_opt else "fallback",
            method=used,
            diagnostics={
                "n_assets": n,
                "uncertainty_type": uset.get("type"),
                "worst_case_return": worst_case_return(w, uset),
                "nominal_return": float(w @ m),
                "variance": portfolio_variance(w, c),
                "rho": uset.get("rho"),
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
