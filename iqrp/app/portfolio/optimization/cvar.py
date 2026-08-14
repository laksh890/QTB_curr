"""Scenario CVaR portfolio optimization (Rockafellar–Uryasev softmin / iterative)."""

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
    project_weights,
    projected_gradient,
    scipy_available,
)


def _softmin_cvar(losses: np.ndarray, alpha: float, temperature: float) -> tuple[float, np.ndarray]:
    """
    Differentiable CVaR approximation via soft tail expectation.

    losses: shape (T,) portfolio losses (= -returns)
    Returns (cvar_approx, d_loss_weights) where gradient is wrt each scenario loss.
    """
    t = losses.size
    if t == 0:
        return 0.0, losses
    max(int(np.ceil((1.0 - alpha) * t)), 1)
    # temperature softmin over the worst k via softmax on losses
    # Use full softplus-style: VaR level via quantile, ES via soft weights above
    q = float(np.quantile(losses, alpha))
    # Soft indicator of tail: sigmoid((loss - q) / temp)
    temp = max(float(temperature), 1e-8)
    logits = (losses - q) / temp
    logits = logits - float(np.max(logits))
    w = np.exp(logits)
    # emphasize upper tail; mix with hard tail for stability
    hard = losses >= q - 1e-15
    if not np.any(hard):
        hard = losses >= float(np.max(losses)) - 1e-15
    mix = 0.7 * (w / max(float(np.sum(w)), 1e-18)) + 0.3 * (
        hard.astype(np.float64) / max(float(np.sum(hard)), 1.0)
    )
    # CVaR ≈ weighted mean of losses in soft tail; also blend VaR
    cvar = float(np.dot(mix, losses))
    # gradient of soft weighted mean wrt losses (approx: mix, ignoring mix dependence)
    return cvar, mix


def optimize_cvar(
    mu: Any = None,
    cov: Any = None,
    *,
    scenarios: Any = None,
    current_weights: Any = None,
    constraints: Any = None,
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_aversion: float = 1.0,
    alpha: float = 0.95,
    temperature: float = 0.01,
    return_tradeoff: float = 0.0,
    min_weight: float | None = None,
    max_gross: float | None = None,
    budget: float = 1.0,
    names: list[str] | None = None,
    max_iter: int = 600,
) -> dict[str, Any]:
    """
    Minimize approximate portfolio CVaR using a scenario return matrix.

    ``scenarios`` is (T, N) historical / simulated returns. If omitted and ``cov``
    is provided, synthetic Gaussian scenarios are drawn for a usable fallback path.
    """
    name = "cvar"
    method = "softmin_cvar"
    try:
        if scenarios is None:
            if cov is None:
                raise ValueError("scenarios or cov required for CVaR optimization")
            c = as_cov(cov)
            n = c.shape[0]
            rng = np.random.default_rng(0)
            # synthetic scenarios for API completeness when only cov given
            mean = np.zeros(n) if mu is None else as_vector(mu, n)
            scenarios_m = rng.multivariate_normal(mean, c, size=max(200, 20 * n))
            synthetic = True
        else:
            scenarios_m = np.asarray(scenarios, dtype=np.float64)
            if scenarios_m.ndim != 2:
                raise ValueError("scenarios must be 2-D (T, N)")
            n = scenarios_m.shape[1]
            synthetic = False
            if cov is not None:
                _ = as_cov(cov, n)

        if scenarios_m.shape[0] < 2:
            raise ValueError("need at least 2 scenarios")

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

        conf = float(alpha)
        if not (0.5 < conf < 1.0):
            raise ValueError("alpha must be in (0.5, 1)")

        m = None if mu is None else as_vector(mu, n)
        lam_ret = float(return_tradeoff)
        # optional mean-CVaR tradeoff using risk_aversion as CVaR weight
        cvar_w = max(float(risk_aversion), 1e-8)

        def project(w: np.ndarray) -> np.ndarray:
            return project_weights(w, cstr)

        if current_weights is not None:
            x0 = project(as_vector(current_weights, n))
        else:
            x0 = project(equal_weights(n, cstr["budget"]))

        def obj(w: np.ndarray) -> float:
            port = scenarios_m @ w
            losses = -port
            cvar, _ = _softmin_cvar(losses, conf, temperature)
            ret_pen = 0.0 if m is None else -lam_ret * float(w @ m)
            return cvar_w * cvar + ret_pen

        def grad(w: np.ndarray) -> np.ndarray:
            port = scenarios_m @ w
            losses = -port
            _, mix = _softmin_cvar(losses, conf, temperature)
            # d loss_t / d w = -scenario_t
            g = cvar_w * (-(mix @ scenarios_m))
            if m is not None and lam_ret != 0.0:
                g = g - lam_ret * m
            return g

        # Iterative reweighted LP-style: alternate softmin weights and quadratic projection
        w = x0.copy()
        used = "iterative_softmin"
        success_opt = True
        iters = 0
        if scipy_available():
            bounds = [(cstr["lb"], cstr["ub"])] * n
            cons = {"type": "eq", "fun": lambda ww: float(np.sum(ww) - cstr["budget"])}
            try:
                res = minimize_scipy(
                    obj, x0, jac=grad, bounds=bounds, constraints=[cons], method="SLSQP"
                )
                if bool(res.success):
                    w = project(np.asarray(res.x, dtype=np.float64))
                    used = "scipy_slsqp_softmin"
                    iters = int(getattr(res, "nit", 0) or 0)
                else:
                    # iterative scenario reweight
                    for it in range(1, max_iter + 1):
                        port = scenarios_m @ w
                        losses = -port
                        _, mix = _softmin_cvar(losses, conf, temperature)
                        # minimize mix @ (-R w) = - (mix R) w  → maximize (mix R) w under constraints
                        score = mix @ scenarios_m
                        # one projected gradient step on -score + tiny ridge
                        w = project(w + 0.1 * score)
                        iters = it
                    used = "iterative_softmin"
            except Exception:
                w, _, success_opt, iters = projected_gradient(
                    obj, grad, x0, project, lr=0.05, max_iter=max_iter
                )
                used = "numpy_pgd"
        else:
            # pure iterative softmin reweighting
            for it in range(1, max_iter + 1):
                port = scenarios_m @ w
                losses = -port
                _, mix = _softmin_cvar(losses, conf, temperature)
                score = mix @ scenarios_m
                w_new = project(w + 0.1 * score)
                if float(np.max(np.abs(w_new - w))) < 1e-10:
                    w = w_new
                    iters = it
                    break
                w = w_new
                iters = it
            # polish with PGD
            w, _, success_opt, it2 = projected_gradient(
                obj, grad, w, project, lr=0.03, max_iter=max(100, max_iter // 3)
            )
            iters += it2
            used = "iterative_softmin"

        if float(np.min(w)) < cstr["lb"] - 1e-8 or float(np.max(w)) > cstr["ub"] + 1e-8:
            return infeasible_result(
                name, n, method=used, reason="box violation", conflicts=["box"], names=names
            )
        if abs(float(np.sum(w)) - cstr["budget"]) > 1e-6:
            return infeasible_result(
                name, n, method=used, reason="budget violation", conflicts=["budget"], names=names
            )

        port = scenarios_m @ w
        losses = -port
        cvar_val, _ = _softmin_cvar(losses, conf, temperature)
        # also report hard historical CVaR
        q = float(np.quantile(losses, conf))
        tail = losses[losses >= q]
        hard_cvar = float(np.mean(tail)) if tail.size else q

        return make_result(
            name,
            format_weights(w, names),
            success=True,
            status="optimal" if success_opt else "fallback",
            method=used,
            diagnostics={
                "n_assets": n,
                "n_scenarios": int(scenarios_m.shape[0]),
                "alpha": conf,
                "soft_cvar": cvar_val,
                "historical_cvar": hard_cvar,
                "var": q,
                "temperature": float(temperature),
                "synthetic_scenarios": synthetic,
                "iterations": iters,
            },
            objective_value=cvar_val,
        )
    except Exception as exc:
        n = 0
        try:
            if scenarios is not None:
                n = int(np.asarray(scenarios).shape[1])
            elif cov is not None:
                n = int(np.asarray(cov).shape[0])
        except Exception:
            n = 0
        return failed_result(name, n, method=method, reason=str(exc))
