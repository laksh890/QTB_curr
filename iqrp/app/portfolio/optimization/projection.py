"""Constraint projection helpers for portfolio optimizers."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
except Exception:  # pragma: no cover
    _scipy_minimize = None


def as_vector(x: Any, n: int | None = None, *, name: str = "vector") -> np.ndarray:
    if x is None:
        if n is None:
            raise ValueError(f"{name} is None and n is unknown")
        return np.zeros(n, dtype=np.float64)
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if n is not None and v.size != n:
        raise ValueError(f"{name} length {v.size} != {n}")
    return v


def as_cov(cov: Any, n: int | None = None) -> np.ndarray:
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square")
    if n is not None and c.shape[0] != n:
        raise ValueError(f"cov shape {c.shape} incompatible with n={n}")
    c = 0.5 * (c + c.T)
    # ridge for numerical stability
    eps = 1e-10 * float(np.trace(c) / max(c.shape[0], 1) + 1e-12)
    c = c + eps * np.eye(c.shape[0])
    return c


def stabilize_mu(mu: np.ndarray, *, clip: float = 0.5, winsor_z: float = 3.0) -> np.ndarray:
    """Winsorize / clip expected returns to avoid extreme MV allocations."""
    m = np.asarray(mu, dtype=np.float64).reshape(-1)
    if m.size == 0:
        return m
    med = float(np.median(m))
    mad = float(np.median(np.abs(m - med))) + 1e-12
    z = (m - med) / (1.4826 * mad)
    m = np.where(np.abs(z) > winsor_z, med + np.sign(z) * winsor_z * 1.4826 * mad, m)
    return np.clip(m, -abs(clip), abs(clip))


def parse_constraints(
    constraints: Any,
    n: int,
    *,
    long_only: bool = True,
    max_weight: float = 0.4,
    min_weight: float | None = None,
    max_gross: float | None = None,
    budget: float = 1.0,
) -> dict[str, Any]:
    """Normalize constraint inputs into a flat dict of hard bounds."""
    lo = bool(long_only)
    ub = float(max_weight) if max_weight is not None else 1.0
    lb = 0.0 if lo else (float(min_weight) if min_weight is not None else -ub)
    if min_weight is not None:
        lb = float(min_weight)
        if lb < 0:
            lo = False
    gross = float(max_gross) if max_gross is not None else None
    bud = float(budget)
    names: list[str] | None = None
    extras: list[dict[str, Any]] = []

    if constraints is None:
        pass
    elif isinstance(constraints, dict):
        if "long_only" in constraints:
            lo = bool(constraints["long_only"])
            if lo and lb < 0:
                lb = 0.0
        if "max_weight" in constraints and constraints["max_weight"] is not None:
            ub = float(constraints["max_weight"])
        if "min_weight" in constraints and constraints["min_weight"] is not None:
            lb = float(constraints["min_weight"])
            if lb < 0:
                lo = False
        if "max_gross" in constraints and constraints["max_gross"] is not None:
            gross = float(constraints["max_gross"])
        if "budget" in constraints and constraints["budget"] is not None:
            bud = float(constraints["budget"])
        elif "sum_weights" in constraints and constraints["sum_weights"] is not None:
            bud = float(constraints["sum_weights"])
        if "names" in constraints:
            names = list(constraints["names"])
        # pass through linear / custom hard constraints for feasibility reporting
        for key in ("linear_eq", "linear_ineq", "group_limits", "sector_limits"):
            if key in constraints and constraints[key] is not None:
                extras.append({key: constraints[key]})
    else:
        # object with attributes
        for attr, setter in (
            ("long_only", "lo"),
            ("max_weight", "ub"),
            ("min_weight", "lb"),
            ("max_gross", "gross"),
            ("budget", "bud"),
        ):
            if hasattr(constraints, attr):
                val = getattr(constraints, attr)
                if val is not None:
                    if setter == "lo":
                        lo = bool(val)
                    elif setter == "ub":
                        ub = float(val)
                    elif setter == "lb":
                        lb = float(val)
                    elif setter == "gross":
                        gross = float(val)
                    elif setter == "bud":
                        bud = float(val)
        if hasattr(constraints, "names") and getattr(constraints, "names") is not None:
            names = list(constraints.names)

    if lo:
        lb = max(lb, 0.0)
    ub = max(ub, lb)
    return {
        "n": n,
        "long_only": lo,
        "lb": lb,
        "ub": ub,
        "max_gross": gross,
        "budget": bud,
        "names": names,
        "extras": extras,
    }


def check_feasibility(cstr: dict[str, Any]) -> tuple[bool, str | None, list[str]]:
    """Hard feasibility checks for box + budget (+ optional gross)."""
    n = int(cstr["n"])
    lb = float(cstr["lb"])
    ub = float(cstr["ub"])
    bud = float(cstr["budget"])
    gross = cstr.get("max_gross")
    conflicts: list[str] = []

    if n <= 0:
        return False, "n_assets must be positive", ["n_assets"]
    if ub < lb:
        conflicts.append("max_weight < min_weight")
    if n * ub + 1e-12 < bud:
        conflicts.append("n * max_weight < budget")
    if n * lb - 1e-12 > bud:
        conflicts.append("n * min_weight > budget")
    if gross is not None:
        g = float(gross)
        if g + 1e-12 < abs(bud):
            conflicts.append("max_gross < |budget|")
        # long-only with nonneg weights: gross == budget when weights sum to budget
        if lb >= 0 and g + 1e-12 < bud:
            conflicts.append("max_gross < budget under long_only")
    if cstr.get("extras"):
        # Unsupported hard extras that cannot be enforced → report conflict rather than ignore silently
        for ex in cstr["extras"]:
            conflicts.append(f"unsupported_hard_constraint:{sorted(ex.keys())[0]}")

    if conflicts:
        return False, "; ".join(conflicts), conflicts
    return True, None, []


def project_simplex(v: np.ndarray, budget: float = 1.0) -> np.ndarray:
    """Project onto {w >= 0, sum w = budget}."""
    x = np.asarray(v, dtype=np.float64).reshape(-1)
    n = x.size
    if n == 0:
        return x
    if budget <= 0:
        return np.zeros(n)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - budget))[0]
    if rho.size == 0:
        theta = 0.0
    else:
        rho_idx = int(rho[-1])
        theta = (cssv[rho_idx] - budget) / float(rho_idx + 1)
    w = np.maximum(x - theta, 0.0)
    s = float(np.sum(w))
    if s <= 0:
        return np.full(n, budget / n)
    return w * (budget / s)


def project_box_simplex(
    v: np.ndarray,
    *,
    lb: float = 0.0,
    ub: float = 1.0,
    budget: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> np.ndarray:
    """Project onto box ∩ affine budget via bisection on the dual."""
    x = np.asarray(v, dtype=np.float64).reshape(-1)
    n = x.size
    if n == 0:
        return x
    if n * ub < budget - 1e-12 or n * lb > budget + 1e-12:
        raise ValueError("box+simplex projection infeasible")

    # Find lambda such that sum(clip(x - lambda, lb, ub)) = budget
    lo, hi = float(np.min(x) - ub - 1.0), float(np.max(x) - lb + 1.0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        w = np.clip(x - mid, lb, ub)
        s = float(np.sum(w))
        if abs(s - budget) < tol:
            return w
        if s > budget:
            lo = mid
        else:
            hi = mid
    return np.clip(x - 0.5 * (lo + hi), lb, ub)


def project_gross(
    w: np.ndarray,
    *,
    max_gross: float,
    budget: float = 1.0,
    long_only: bool = False,
) -> np.ndarray:
    """Scale to satisfy gross exposure if needed (preserves sign pattern)."""
    x = np.asarray(w, dtype=np.float64).reshape(-1)
    if long_only:
        x = np.maximum(x, 0.0)
    g = float(np.sum(np.abs(x)))
    if g <= max_gross + 1e-12:
        # still enforce budget if long_only / sum constraint desired
        s = float(np.sum(x))
        if abs(s - budget) > 1e-10 and abs(s) > 1e-14:
            x = x * (budget / s)
        return x
    # shrink gross
    x = x * (max_gross / g)
    s = float(np.sum(x))
    if abs(s) > 1e-14 and abs(budget) > 1e-14:
        # try to restore budget without exceeding gross: not always possible
        scale = budget / s
        trial = x * scale
        if float(np.sum(np.abs(trial))) <= max_gross + 1e-10:
            return trial
    return x


def project_weights(
    v: np.ndarray,
    cstr: dict[str, Any],
) -> np.ndarray:
    """Project onto configured hard constraints (box + budget + optional gross)."""
    lb = float(cstr["lb"])
    ub = float(cstr["ub"])
    bud = float(cstr["budget"])
    long_only = bool(cstr["long_only"])
    x = np.asarray(v, dtype=np.float64).reshape(-1)

    if long_only or lb >= 0:
        w = project_box_simplex(x, lb=max(lb, 0.0), ub=ub, budget=bud)
    else:
        # long-short: project to budget via shift then box-clip and renormalize budget
        w = x - (float(np.sum(x)) - bud) / max(x.size, 1)
        w = np.clip(w, lb, ub)
        # restore budget with minimal L2 change under box via dual bisection
        w = project_box_simplex(w, lb=lb, ub=ub, budget=bud) if lb >= 0 else _project_budget_box_ls(w, lb, ub, bud)

    gross = cstr.get("max_gross")
    if gross is not None:
        w = project_gross(w, max_gross=float(gross), budget=bud, long_only=long_only or lb >= 0)
        # re-apply box after gross scale
        w = np.clip(w, lb, ub)
        if long_only or lb >= 0:
            s = float(np.sum(w))
            if s > 1e-14:
                w = w * (bud / s)
                w = np.clip(w, lb, ub)
                s2 = float(np.sum(w))
                if abs(s2 - bud) > 1e-8 and s2 > 0:
                    w = project_box_simplex(w, lb=max(lb, 0.0), ub=ub, budget=bud)
    return w


def _project_budget_box_ls(x: np.ndarray, lb: float, ub: float, budget: float) -> np.ndarray:
    lo, hi = float(np.min(x) - ub - 1.0), float(np.max(x) - lb + 1.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        w = np.clip(x - mid, lb, ub)
        s = float(np.sum(w))
        if abs(s - budget) < 1e-12:
            return w
        if s > budget:
            lo = mid
        else:
            hi = mid
    return np.clip(x - 0.5 * (lo + hi), lb, ub)


def equal_weights(n: int, budget: float = 1.0) -> np.ndarray:
    if n <= 0:
        return np.zeros(0)
    return np.full(n, budget / n, dtype=np.float64)


def portfolio_variance(w: np.ndarray, cov: np.ndarray) -> float:
    return float(w @ cov @ w)


def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def make_result(
    name: str,
    weights: np.ndarray | list[float] | dict[str, float],
    *,
    success: bool,
    status: str,
    method: str,
    failure_reason: str | None = None,
    conflicting_constraints: list[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
    objective_value: float | None = None,
) -> dict[str, Any]:
    if isinstance(weights, np.ndarray):
        w_out: list[float] | dict[str, float] = [float(x) for x in weights.tolist()]
    else:
        w_out = weights
    return {
        "name": name,
        "success": bool(success),
        "weights": w_out,
        "status": status,
        "failure_reason": failure_reason,
        "conflicting_constraints": list(conflicting_constraints or []),
        "diagnostics": dict(diagnostics or {}),
        "objective_value": None if objective_value is None else float(objective_value),
        "method": method,
    }


def infeasible_result(
    name: str,
    n: int,
    *,
    method: str,
    reason: str,
    conflicts: list[str],
    names: list[str] | None = None,
) -> dict[str, Any]:
    if names and len(names) == n:
        w: list[float] | dict[str, float] = {names[i]: 0.0 for i in range(n)}
    else:
        w = [0.0] * n
    return make_result(
        name,
        w,
        success=False,
        status="infeasible",
        method=method,
        failure_reason=reason,
        conflicting_constraints=conflicts,
        diagnostics={"n_assets": n},
        objective_value=None,
    )


def failed_result(
    name: str,
    n: int,
    *,
    method: str,
    reason: str,
    names: list[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if names and len(names) == n:
        w: list[float] | dict[str, float] = {names[i]: 0.0 for i in range(n)}
    else:
        w = [0.0] * n
    return make_result(
        name,
        w,
        success=False,
        status="failed",
        method=method,
        failure_reason=reason,
        diagnostics=diagnostics or {"n_assets": n},
    )


def format_weights(w: np.ndarray, names: list[str] | None) -> list[float] | dict[str, float]:
    if names and len(names) == w.size:
        return {names[i]: float(w[i]) for i in range(w.size)}
    return [float(x) for x in w.tolist()]


def scipy_available() -> bool:
    return _scipy_minimize is not None


def minimize_scipy(
    fun,
    x0: np.ndarray,
    *,
    jac=None,
    bounds=None,
    constraints=None,
    method: str = "SLSQP",
    options: dict[str, Any] | None = None,
):
    if _scipy_minimize is None:
        raise RuntimeError("scipy.optimize unavailable")
    return _scipy_minimize(
        fun,
        x0,
        jac=jac,
        bounds=bounds,
        constraints=constraints or (),
        method=method,
        options=options or {"maxiter": 500, "ftol": 1e-12},
    )


def projected_gradient(
    fun,
    grad,
    x0: np.ndarray,
    project,
    *,
    lr: float = 0.05,
    max_iter: int = 800,
    tol: float = 1e-10,
    lr_decay: float = 0.999,
) -> tuple[np.ndarray, float, bool, int]:
    x = project(np.asarray(x0, dtype=np.float64))
    f = float(fun(x))
    best_x, best_f = x.copy(), f
    step = lr
    for it in range(1, max_iter + 1):
        g = np.asarray(grad(x), dtype=np.float64)
        x_new = project(x - step * g)
        f_new = float(fun(x_new))
        if f_new < best_f:
            best_f, best_x = f_new, x_new.copy()
        if float(np.max(np.abs(x_new - x))) < tol:
            return x_new, f_new, True, it
        x, f = x_new, f_new
        step *= lr_decay
    return best_x, best_f, False, max_iter
