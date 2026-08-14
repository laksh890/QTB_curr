"""Maximum-entropy / diversity-regularized mean-variance optimization."""

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


def _entropy(w: np.ndarray) -> float:
    x = np.clip(w, 1e-16, None)
    x = x / max(float(np.sum(x)), 1e-16)
    return float(-np.sum(x * np.log(x)))


def optimize_entropy(
    mu: Any = None,
    cov: Any = None,
    *,
    current_weights: Any = None,
    constraints: Any = None,
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_aversion: float = 1.0,
    entropy_weight: float = 0.1,
    min_weight: float | None = None,
    max_gross: float | None = None,
    budget: float = 1.0,
    ridge: float = 1e-8,
    names: list[str] | None = None,
    mu_clip: float = 0.5,
) -> dict[str, Any]:
    """
    Maximize w'μ - (λ/2) w'Σw + γ H(w) where H is Shannon entropy of weights.

    Long-only is required for a well-defined probability-simplex entropy.
    """
    name = "entropy"
    method = "entropy_regularized_mv"
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
        if cstr["lb"] < 0:
            return infeasible_result(
                name,
                n,
                method=method,
                reason="entropy regularization requires non-negative weights",
                conflicts=["long_only"],
                names=names or cstr.get("names"),
            )
        cstr["long_only"] = True
        cstr["lb"] = max(float(cstr["lb"]), 0.0)
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

        lam = max(float(risk_aversion), 1e-8)
        gamma = float(entropy_weight)

        def project(w: np.ndarray) -> np.ndarray:
            return project_weights(w, cstr)

        x0 = project(
            as_vector(current_weights, n)
            if current_weights is not None
            else equal_weights(n, cstr["budget"])
        )

        def obj(w: np.ndarray) -> float:
            # minimize negative utility
            p = np.clip(w, 1e-16, None)
            p = p / max(float(np.sum(p)), 1e-16)
            ent = -float(np.sum(p * np.log(p)))
            return 0.5 * lam * portfolio_variance(w, c) - portfolio_return(w, m) - gamma * ent

        def grad(w: np.ndarray) -> np.ndarray:
            p = np.clip(w, 1e-16, None)
            s = max(float(np.sum(p)), 1e-16)
            p = p / s
            # dH/dw_i ≈ -(log p_i + 1)/s  (simplex-normalized)
            d_ent = -(np.log(p) + 1.0) / s
            return lam * (c @ w) - m - gamma * d_ent

        used = "numpy_pgd"
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
                    success_opt = True
                    iters = int(getattr(res, "nit", 0) or 0)
                else:
                    w, fval, success_opt, iters = projected_gradient(
                        obj, grad, x0, project, lr=0.05
                    )
            except Exception:
                w, fval, success_opt, iters = projected_gradient(obj, grad, x0, project, lr=0.05)
        else:
            w, fval, success_opt, iters = projected_gradient(obj, grad, x0, project, lr=0.05)

        if float(np.min(w)) < cstr["lb"] - 1e-8 or float(np.max(w)) > cstr["ub"] + 1e-8:
            return infeasible_result(
                name, n, method=used, reason="box violation", conflicts=["box"], names=names
            )
        if abs(float(np.sum(w)) - cstr["budget"]) > 1e-6:
            return infeasible_result(
                name, n, method=used, reason="budget violation", conflicts=["budget"], names=names
            )

        ent = _entropy(w)
        return make_result(
            name,
            format_weights(w, names),
            success=True,
            status="optimal" if success_opt else "fallback",
            method=used,
            diagnostics={
                "n_assets": n,
                "entropy": ent,
                "max_entropy": float(np.log(n)),
                "entropy_weight": gamma,
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
