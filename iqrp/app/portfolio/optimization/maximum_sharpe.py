"""Maximum Sharpe (tangency) portfolio optimization."""

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


def optimize_maximum_sharpe(
    mu: Any = None,
    cov: Any = None,
    *,
    current_weights: Any = None,
    constraints: Any = None,
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
    min_weight: float | None = None,
    max_gross: float | None = None,
    budget: float = 1.0,
    ridge: float = 1e-8,
    names: list[str] | None = None,
    mu_clip: float = 0.5,
) -> dict[str, Any]:
    """
    Tangency portfolio maximizing (w'μ - rf) / sqrt(w'Σw) with hard constraints.

    Uses excess returns and a numerically stable negative-Sharpe minimization.
    """
    name = "maximum_sharpe"
    method = "tangency"
    _ = risk_aversion
    try:
        if cov is None:
            raise ValueError("cov is required")
        if mu is None:
            raise ValueError("mu is required for maximum Sharpe")
        c = as_cov(cov)
        n = c.shape[0]
        c = c + float(ridge) * np.eye(n)
        m = stabilize_mu(as_vector(mu, n, name="mu"), clip=mu_clip)
        excess = m - float(risk_free_rate)

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

        # Analytic tangency (unconstrained), then project
        try:
            raw = np.linalg.solve(c, excess)
            if float(np.sum(np.abs(raw))) < 1e-14:
                raw = equal_weights(n, 1.0)
            if cstr["long_only"] or cstr["lb"] >= 0:
                raw = np.maximum(raw, 0.0)
                if float(np.sum(raw)) <= 1e-14:
                    raw = equal_weights(n, 1.0)
            w_tan = raw / float(np.sum(raw)) * cstr["budget"]
        except np.linalg.LinAlgError:
            w_tan = equal_weights(n, cstr["budget"])

        def project(w: np.ndarray) -> np.ndarray:
            return project_weights(w, cstr)

        x0 = project(as_vector(current_weights, n) if current_weights is not None else w_tan)

        def neg_sharpe(w: np.ndarray) -> float:
            er = portfolio_return(w, m) - float(risk_free_rate)
            vol = float(np.sqrt(max(portfolio_variance(w, c), 1e-18)))
            return -er / vol

        def grad_ns(w: np.ndarray) -> np.ndarray:
            # numerical-friendly analytic gradient of - (ex / vol)
            ex = portfolio_return(w, m) - float(risk_free_rate)
            var = max(portfolio_variance(w, c), 1e-18)
            vol = float(np.sqrt(var))
            d_ex = m
            d_var = 2.0 * (c @ w)
            d_vol = d_var / (2.0 * vol)
            # d(ex/vol)/dw = (d_ex * vol - ex * d_vol) / vol^2
            d_sharpe = (d_ex * vol - ex * d_vol) / (vol * vol)
            return -d_sharpe

        used = "analytic_tangency_projection"
        w = project(w_tan)
        fval = float(neg_sharpe(w))
        success_opt = True
        iters = 0

        if scipy_available():
            bounds = [(cstr["lb"], cstr["ub"])] * n
            cons = {"type": "eq", "fun": lambda ww: float(np.sum(ww) - cstr["budget"])}
            try:
                res = minimize_scipy(
                    neg_sharpe, x0, jac=grad_ns, bounds=bounds, constraints=[cons], method="SLSQP"
                )
                if bool(res.success):
                    w = project(np.asarray(res.x, dtype=np.float64))
                    fval = float(neg_sharpe(w))
                    used = "scipy_slsqp"
                    iters = int(getattr(res, "nit", 0) or 0)
                else:
                    w, fval, success_opt, iters = projected_gradient(
                        neg_sharpe, grad_ns, x0, project, lr=0.05
                    )
                    used = "numpy_pgd"
            except Exception:
                w, fval, success_opt, iters = projected_gradient(
                    neg_sharpe, grad_ns, x0, project, lr=0.05
                )
                used = "numpy_pgd"
        else:
            w, fval, success_opt, iters = projected_gradient(
                neg_sharpe, grad_ns, x0, project, lr=0.05
            )
            used = "numpy_pgd"

        if float(np.min(w)) < cstr["lb"] - 1e-8 or float(np.max(w)) > cstr["ub"] + 1e-8:
            return infeasible_result(
                name, n, method=used, reason="box violation", conflicts=["box"], names=names
            )
        if abs(float(np.sum(w)) - cstr["budget"]) > 1e-6:
            return infeasible_result(
                name, n, method=used, reason="budget violation", conflicts=["budget"], names=names
            )

        er = portfolio_return(w, m)
        vol = float(np.sqrt(max(portfolio_variance(w, c), 0.0)))
        sharpe = (er - float(risk_free_rate)) / max(vol, 1e-18)
        return make_result(
            name,
            format_weights(w, names),
            success=True,
            status="optimal" if success_opt else "fallback",
            method=used,
            diagnostics={
                "n_assets": n,
                "risk_free_rate": float(risk_free_rate),
                "expected_return": er,
                "volatility": vol,
                "sharpe": sharpe,
                "iterations": iters,
            },
            objective_value=sharpe,
        )
    except Exception as exc:
        n = 0
        try:
            n = int(np.asarray(cov).shape[0]) if cov is not None else 0
        except Exception:
            n = 0
        return failed_result(name, n, method=method, reason=str(exc))
