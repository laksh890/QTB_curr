"""Risk-parity optimizer wrapping existing risk sizing implementations."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.portfolio.optimization.projection import (
    as_cov,
    check_feasibility,
    failed_result,
    format_weights,
    infeasible_result,
    make_result,
    parse_constraints,
    project_weights,
)
from iqrp.app.risk.sizing.risk_parity import equal_risk_contribution, risk_parity_weights


def optimize_risk_parity(
    mu: Any = None,
    cov: Any = None,
    *,
    current_weights: Any = None,
    constraints: Any = None,
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_aversion: float = 1.0,
    method: str = "risk_parity",
    min_weight: float | None = None,
    max_gross: float | None = None,
    budget: float = 1.0,
    names: list[str] | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """
    Equal-risk-contribution / risk-parity weights via iqrp.app.risk.sizing.

    Does not reimplement the ERC solver. Applies hard box/budget projection after;
    if the projected solution cannot satisfy hard constraints, returns infeasible.
    """
    name = "risk_parity"
    _ = mu
    _ = risk_aversion
    _ = current_weights
    try:
        if cov is None:
            raise ValueError("cov is required")
        c = as_cov(cov)
        n = c.shape[0]

        cstr = parse_constraints(
            constraints,
            n,
            long_only=long_only,
            max_weight=max_weight,
            min_weight=min_weight,
            max_gross=max_gross,
            budget=budget,
        )
        # Risk parity is inherently long-only; reject short-capable hard specs
        if not cstr["long_only"] and cstr["lb"] < 0:
            return infeasible_result(
                name,
                n,
                method="risk_parity_weights",
                reason="risk parity requires long_only / non-negative weights",
                conflicts=["long_only"],
                names=names or cstr.get("names"),
            )
        cstr["long_only"] = True
        cstr["lb"] = max(float(cstr["lb"]), 0.0)

        if names is None:
            names = cstr.get("names")
        ok, reason, conflicts = check_feasibility(cstr)
        if not ok:
            return infeasible_result(name, n, method="risk_parity_weights", reason=reason or "infeasible", conflicts=conflicts, names=names)

        m = str(method).lower()
        if m in {"erc", "equal_risk_contribution", "equal_risk"}:
            raw = equal_risk_contribution(c, max_iter=max_iter, tol=tol)
            used = "equal_risk_contribution"
        else:
            raw = risk_parity_weights(c, max_iter=max_iter, tol=tol)
            used = "risk_parity_weights"

        w = np.asarray(raw.get("weights", []), dtype=np.float64).reshape(-1)
        if w.size != n:
            raise ValueError("risk parity backend returned unexpected weight size")
        w = w * float(cstr["budget"])
        w = project_weights(w, cstr)

        if float(np.min(w)) < cstr["lb"] - 1e-8 or float(np.max(w)) > cstr["ub"] + 1e-8:
            return infeasible_result(
                name,
                n,
                method=used,
                reason="projected risk-parity weights violate box constraints",
                conflicts=["box"],
                names=names,
            )
        if abs(float(np.sum(w)) - cstr["budget"]) > 1e-6:
            return infeasible_result(
                name,
                n,
                method=used,
                reason="projected risk-parity weights violate budget",
                conflicts=["budget"],
                names=names,
            )

        diag = {
            "n_assets": n,
            "backend": used,
            "converged": bool(raw.get("converged", True)),
            "iterations": raw.get("iterations"),
        }
        if "component_risk_contribution" in raw:
            diag["component_risk_contribution"] = raw["component_risk_contribution"]

        return make_result(
            name,
            format_weights(w, names),
            success=True,
            status="optimal" if diag["converged"] else "fallback",
            method=used,
            diagnostics=diag,
            objective_value=None,
        )
    except Exception as exc:
        n = 0
        try:
            n = int(np.asarray(cov).shape[0]) if cov is not None else 0
        except Exception:
            n = 0
        return failed_result(name, n, method="risk_parity_weights", reason=str(exc))
