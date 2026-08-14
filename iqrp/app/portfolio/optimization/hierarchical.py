"""Hierarchical risk parity / HERC optimizers wrapping capital.hierarchical."""

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
from iqrp.app.risk.capital.hierarchical import herc_weights, hrp_weights


def optimize_hierarchical(
    mu: Any = None,
    cov: Any = None,
    *,
    current_weights: Any = None,
    constraints: Any = None,
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_aversion: float = 1.0,
    variant: str = "hrp",
    corr: Any = None,
    linkage: str = "single",
    min_weight: float | None = None,
    max_gross: float | None = None,
    budget: float = 1.0,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """HRP / HERC via iqrp.app.risk.capital.hierarchical (no reimplementation)."""
    name = "hierarchical"
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
        if not cstr["long_only"] and cstr["lb"] < 0:
            return infeasible_result(
                name,
                n,
                method=str(variant),
                reason="hierarchical methods require non-negative weights",
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
                method=str(variant),
                reason=reason or "infeasible",
                conflicts=conflicts,
                names=names,
            )

        key_names = names if names and len(names) == n else [f"a{i}" for i in range(n)]
        v = str(variant).lower()
        if v in {"herc", "hierarchical_erc", "equal_risk"}:
            raw = herc_weights(c, names=key_names, corr=corr, linkage=linkage)
            used = "herc_weights"
        else:
            raw = hrp_weights(c, names=key_names, corr=corr, linkage=linkage)
            used = "hrp_weights"

        if "weight_vector" in raw:
            w = np.asarray(raw["weight_vector"], dtype=np.float64).reshape(-1)
        else:
            wd = raw.get("weights", {})
            w = np.asarray([float(wd[k]) for k in key_names], dtype=np.float64)
        if w.size != n:
            raise ValueError("hierarchical backend returned unexpected weight size")
        w = w * float(cstr["budget"])
        w = project_weights(w, cstr)

        if float(np.min(w)) < cstr["lb"] - 1e-8 or float(np.max(w)) > cstr["ub"] + 1e-8:
            return infeasible_result(
                name,
                n,
                method=used,
                reason="projected hierarchical weights violate box constraints",
                conflicts=["box"],
                names=names,
            )
        if abs(float(np.sum(w)) - cstr["budget"]) > 1e-6:
            return infeasible_result(
                name,
                n,
                method=used,
                reason="projected hierarchical weights violate budget",
                conflicts=["budget"],
                names=names,
            )

        return make_result(
            name,
            format_weights(w, names),
            success=True,
            status="optimal",
            method=used,
            diagnostics={
                "n_assets": n,
                "variant": v,
                "order": raw.get("order"),
                "linkage_method": raw.get("linkage_method"),
                "linkage_backend": raw.get("linkage_backend"),
            },
            objective_value=None,
        )
    except Exception as exc:
        n = 0
        try:
            n = int(np.asarray(cov).shape[0]) if cov is not None else 0
        except Exception:
            n = 0
        return failed_result(name, n, method=str(variant), reason=str(exc))


def optimize_hrp(
    mu: Any = None,
    cov: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return optimize_hierarchical(mu=mu, cov=cov, variant="hrp", **kwargs)


def optimize_herc(
    mu: Any = None,
    cov: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return optimize_hierarchical(mu=mu, cov=cov, variant="herc", **kwargs)
