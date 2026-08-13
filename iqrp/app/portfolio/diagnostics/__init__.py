"""Portfolio construction diagnostics: numerical health, feasibility, diversification."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.portfolio.constraints.concentration import concentration_metrics
from iqrp.app.portfolio.constraints.exposure import exposure_metrics


def numerical_health(
    *,
    weights: Any | None = None,
    cov: Any | None = None,
    mu: Any | None = None,
) -> dict[str, Any]:
    """Check finiteness, PSD covariance, and extreme weight magnitudes."""
    issues: list[str] = []
    checks: dict[str, Any] = {}

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        checks["weights"] = {
            "n": int(w.size),
            "sum": float(np.sum(w)) if w.size else 0.0,
            "l1": float(np.sum(np.abs(w))) if w.size else 0.0,
            "n_nonfinite": int(np.sum(~np.isfinite(w))) if w.size else 0,
            "max_abs": float(np.max(np.abs(w))) if w.size else 0.0,
        }
        if checks["weights"]["n_nonfinite"] > 0:
            issues.append("weights_nonfinite")
        if w.size and checks["weights"]["max_abs"] > 10.0:
            issues.append("weights_extreme")

    if mu is not None:
        m = np.asarray(mu, dtype=np.float64).reshape(-1)
        checks["mu"] = {
            "n": int(m.size),
            "n_nonfinite": int(np.sum(~np.isfinite(m))) if m.size else 0,
            "mean": float(np.mean(m[np.isfinite(m)])) if m.size and np.any(np.isfinite(m)) else 0.0,
        }
        if checks["mu"]["n_nonfinite"] > 0:
            issues.append("mu_nonfinite")

    if cov is not None:
        c = np.asarray(cov, dtype=np.float64)
        ok = c.ndim == 2 and c.shape[0] == c.shape[1]
        checks["cov"] = {"shape": list(c.shape), "square": ok}
        if not ok:
            issues.append("cov_not_square")
        else:
            sym_err = float(np.max(np.abs(c - c.T))) if c.size else 0.0
            eig = np.linalg.eigvalsh(0.5 * (c + c.T)) if c.size else np.array([])
            min_eig = float(np.min(eig)) if eig.size else 0.0
            cond = float(np.linalg.cond(c)) if c.size else float("inf")
            checks["cov"].update(
                {
                    "symmetry_error": sym_err,
                    "min_eigenvalue": min_eig,
                    "condition_number": cond,
                    "n_negative_eigenvalues": int(np.sum(eig < -1e-10)) if eig.size else 0,
                }
            )
            if sym_err > 1e-8:
                issues.append("cov_asymmetric")
            if min_eig < -1e-8:
                issues.append("cov_not_psd")
            if cond > 1e12:
                issues.append("cov_ill_conditioned")

    healthy = len(issues) == 0
    score = 1.0 if healthy else float(np.clip(1.0 - 0.15 * len(issues), 0.0, 1.0))
    return {
        "name": "numerical_health",
        "healthy": healthy,
        "score": score,
        "issues": issues,
        "checks": checks,
    }


def feasibility_diagnostics(
    weights: Any,
    *,
    max_weight: float | None = None,
    max_gross: float | None = None,
    max_leverage: float | None = None,
    long_only: bool = False,
    budget: float | None = 1.0,
    budget_tol: float = 1e-6,
) -> dict[str, Any]:
    """Assess whether weights satisfy common hard construction constraints."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    exp = exposure_metrics(w)
    conc = concentration_metrics(w)
    violations: list[str] = []

    if long_only and w.size and float(np.min(w)) < -1e-12:
        violations.append("long_only")
    if max_weight is not None and conc["max_weight"] > float(max_weight) + 1e-12:
        violations.append("max_weight")
    if max_gross is not None and exp["gross"] > float(max_gross) + 1e-12:
        violations.append("max_gross")
    if max_leverage is not None and exp["gross"] > float(max_leverage) + 1e-12:
        violations.append("max_leverage")
    if budget is not None and abs(exp["net"] - float(budget)) > float(budget_tol):
        violations.append("budget")

    return {
        "name": "feasibility_diagnostics",
        "feasible": len(violations) == 0,
        "violations": violations,
        "exposure": exp,
        "concentration": conc,
        "parameters": {
            "max_weight": max_weight,
            "max_gross": max_gross,
            "max_leverage": max_leverage,
            "long_only": long_only,
            "budget": budget,
            "budget_tol": budget_tol,
        },
    }


def diversification_metrics(
    weights: Any,
    cov: Any | None = None,
) -> dict[str, Any]:
    """Diversification scorecard: effective N, HHI, optional diversification ratio."""
    conc = concentration_metrics(weights)
    out: dict[str, Any] = {
        "name": "diversification_metrics",
        "hhi": conc["hhi"],
        "effective_n": conc["effective_n"],
        "max_weight": conc["max_weight"],
        "n_assets": conc.get("n_assets", 0.0),
    }
    if cov is not None:
        from iqrp.app.risk.portfolio.diversification import diversification_ratio

        dr = diversification_ratio(weights, cov)
        out["diversification_ratio"] = float(dr.value)
        out["diversification_ratio_detail"] = dr.to_dict()
    # Score in [0,1]: higher = more diversified
    n = max(int(conc.get("n_assets", 1) or 1), 1)
    min_hhi = 1.0 / n
    hhi = conc["hhi"]
    out["score"] = float(np.clip((1.0 - hhi) / max(1.0 - min_hhi, 1e-12), 0.0, 1.0))
    return out


def portfolio_diagnostics(
    weights: Any,
    *,
    cov: Any | None = None,
    mu: Any | None = None,
    max_weight: float | None = None,
    max_gross: float | None = None,
    long_only: bool = False,
) -> dict[str, Any]:
    """Aggregate numerical + feasibility + diversification diagnostics."""
    return {
        "name": "portfolio_diagnostics",
        "numerical_health": numerical_health(weights=weights, cov=cov, mu=mu),
        "feasibility": feasibility_diagnostics(
            weights,
            max_weight=max_weight,
            max_gross=max_gross,
            long_only=long_only,
        ),
        "diversification": diversification_metrics(weights, cov),
    }
