"""Drawdown-aware portfolio optimization via historical DD proxy shrinkage."""

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
    project_weights,
)


def _max_drawdown(path: np.ndarray) -> float:
    if path.size == 0:
        return 0.0
    peak = np.maximum.accumulate(path)
    dd = 1.0 - path / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def _asset_drawdown_scales(
    returns: np.ndarray,
    *,
    soft_cap: float = 0.20,
) -> np.ndarray:
    """Per-asset scale in (0, 1] from historical path max drawdown."""
    t, n = returns.shape
    scales = np.ones(n, dtype=np.float64)
    for i in range(n):
        wealth = np.cumprod(1.0 + returns[:, i])
        mdd = _max_drawdown(wealth)
        scales[i] = float(np.clip(1.0 - mdd / max(soft_cap, 1e-12), 0.05, 1.0))
    return scales


def optimize_drawdown(
    mu: Any = None,
    cov: Any = None,
    *,
    returns: Any = None,
    current_weights: Any = None,
    constraints: Any = None,
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_aversion: float = 1.0,
    drawdown_cap: float = 0.20,
    path_penalty: float = 1.0,
    min_weight: float | None = None,
    max_gross: float | None = None,
    budget: float = 1.0,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Drawdown-aware allocation:

    1) Solve a base mean-variance problem (or equal weight if mu missing).
    2) Shrink weights by per-asset historical drawdown scales and/or portfolio path DD.
    3) Re-project onto hard constraints (never silently relax).
    """
    name = "drawdown"
    method = "dd_shrink_mv"
    try:
        if cov is None and returns is None:
            raise ValueError("cov or returns required")

        if returns is not None:
            r = np.asarray(returns, dtype=np.float64)
            if r.ndim != 2:
                raise ValueError("returns must be (T, N)")
            n = r.shape[1]
            if cov is None:
                c = as_cov(np.cov(r, rowvar=False))
            else:
                c = as_cov(cov, n)
        else:
            c = as_cov(cov)
            n = c.shape[0]
            # synthetic path from cov for DD proxy
            rng = np.random.default_rng(1)
            mean = np.zeros(n) if mu is None else as_vector(mu, n)
            r = rng.multivariate_normal(mean, c, size=max(120, 25 * n))

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

        base = optimize_mean_variance(
            mu=mu if mu is not None else np.zeros(n),
            cov=c,
            current_weights=current_weights,
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
            names=None,
        )
        if not base.get("success"):
            # fall back to equal weight seed
            w0 = equal_weights(n, cstr["budget"])
            if current_weights is not None:
                w0 = as_vector(current_weights, n)
        else:
            w0 = as_vector(base["weights"], n)

        scales = _asset_drawdown_scales(r, soft_cap=drawdown_cap)
        w = w0 * (scales ** float(path_penalty))
        w = project_weights(w, cstr)

        # Portfolio path DD diagnostic and optional global shrink
        port_rets = r @ w
        wealth = np.cumprod(1.0 + port_rets)
        port_mdd = _max_drawdown(wealth)
        if port_mdd > drawdown_cap:
            # shrink active risk toward equal weight without relaxing box/budget
            ew = equal_weights(n, cstr["budget"])
            blend = float(np.clip(drawdown_cap / max(port_mdd, 1e-12), 0.0, 1.0))
            w = project_weights(blend * w + (1.0 - blend) * ew, cstr)
            method = "dd_shrink_mv_blend"
            port_rets = r @ w
            wealth = np.cumprod(1.0 + port_rets)
            port_mdd = _max_drawdown(wealth)

        if float(np.min(w)) < cstr["lb"] - 1e-8 or float(np.max(w)) > cstr["ub"] + 1e-8:
            return infeasible_result(
                name, n, method=method, reason="box violation", conflicts=["box"], names=names
            )
        if abs(float(np.sum(w)) - cstr["budget"]) > 1e-6:
            return infeasible_result(
                name, n, method=method, reason="budget violation", conflicts=["budget"], names=names
            )

        return make_result(
            name,
            format_weights(w, names),
            success=True,
            status="optimal",
            method=method,
            diagnostics={
                "n_assets": n,
                "asset_dd_scales": scales.tolist(),
                "portfolio_max_drawdown": port_mdd,
                "drawdown_cap": float(drawdown_cap),
                "base_method": base.get("method"),
                "base_success": bool(base.get("success")),
            },
            objective_value=port_mdd,
        )
    except Exception as exc:
        n = 0
        try:
            if returns is not None:
                n = int(np.asarray(returns).shape[1])
            elif cov is not None:
                n = int(np.asarray(cov).shape[0])
        except Exception:
            n = 0
        return failed_result(name, n, method=method, reason=str(exc))
