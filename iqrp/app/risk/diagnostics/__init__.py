"""Numerical health diagnostics for risk inputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_returns, as_weights


def risk_diagnostics(
    *,
    returns: Any | None = None,
    weights: Any | None = None,
    cov: Any | None = None,
) -> dict[str, Any]:
    """Summarize numerical health of core risk inputs."""
    issues: list[str] = []
    checks: dict[str, Any] = {}

    if returns is not None:
        r = np.asarray(returns, dtype=np.float64)
        flat = as_returns(r)
        n_nan = int(np.sum(~np.isfinite(r))) if r.size else 0
        checks["returns"] = {
            "n_obs": int(flat.size),
            "n_nonfinite": n_nan,
            "mean": float(np.mean(flat)) if flat.size else 0.0,
            "std": float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0,
            "min": float(np.min(flat)) if flat.size else 0.0,
            "max": float(np.max(flat)) if flat.size else 0.0,
        }
        if flat.size < 2:
            issues.append("returns_insufficient_obs")
        if n_nan > 0:
            issues.append("returns_nonfinite")
        if flat.size > 1 and checks["returns"]["std"] == 0.0:
            issues.append("returns_zero_variance")

    if weights is not None:
        w = as_weights(weights)
        checks["weights"] = {
            "n": int(w.size),
            "sum": float(np.sum(w)),
            "l1": float(np.sum(np.abs(w))),
            "max_abs": float(np.max(np.abs(w))) if w.size else 0.0,
            "n_nonfinite": int(np.sum(~np.isfinite(w))),
        }
        if checks["weights"]["n_nonfinite"] > 0:
            issues.append("weights_nonfinite")
        if w.size and abs(checks["weights"]["sum"]) > 5.0:
            issues.append("weights_extreme_net")

    if cov is not None:
        c = np.asarray(cov, dtype=np.float64)
        ok_shape = c.ndim == 2 and c.shape[0] == c.shape[1]
        checks["cov"] = {"shape": list(c.shape), "square": ok_shape}
        if not ok_shape:
            issues.append("cov_not_square")
        else:
            sym_err = float(np.max(np.abs(c - c.T))) if c.size else 0.0
            eig = np.linalg.eigvalsh(0.5 * (c + c.T)) if c.size else np.array([])
            min_eig = float(np.min(eig)) if eig.size else 0.0
            checks["cov"].update(
                {
                    "symmetry_error": sym_err,
                    "min_eigenvalue": min_eig,
                    "n_negative_eigenvalues": int(np.sum(eig < -1e-10)) if eig.size else 0,
                    "condition_number": (
                        float(np.linalg.cond(c))
                        if c.size and np.linalg.det(c) != 0
                        else float("inf")
                    ),
                }
            )
            if sym_err > 1e-8:
                issues.append("cov_asymmetric")
            if min_eig < -1e-8:
                issues.append("cov_not_psd")
            if checks["cov"]["condition_number"] > 1e12:
                issues.append("cov_ill_conditioned")

    healthy = len(issues) == 0
    score = 1.0 if healthy else float(np.clip(1.0 - 0.15 * len(issues), 0.0, 1.0))

    return {
        "name": "risk_diagnostics",
        "healthy": healthy,
        "issues": issues,
        "checks": checks,
        "measure": RiskMeasure(
            name="numerical_health",
            value=score,
            unit="score",
            method="risk_diagnostics",
            parameters={"n_issues": len(issues)},
            metadata={"issues": list(issues)},
        ).to_dict(),
    }
