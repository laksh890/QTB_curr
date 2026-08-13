"""Numerical health checks for covariance and capital weights."""

from __future__ import annotations

from typing import Any

import numpy as np


def diagnose_covariance(cov: Any) -> dict[str, Any]:
    """Assess covariance matrix numerical health."""
    c = np.asarray(cov, dtype=np.float64)
    issues: list[str] = []
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        return {
            "name": "diagnose_covariance",
            "ok": False,
            "issues": ["not_square"],
            "shape": list(c.shape),
        }
    n = c.shape[0]
    if n == 0:
        return {"name": "diagnose_covariance", "ok": True, "issues": [], "shape": [0, 0]}

    if not np.all(np.isfinite(c)):
        issues.append("non_finite")
    asym = float(np.max(np.abs(c - c.T)))
    if asym > 1e-8:
        issues.append("asymmetric")
    try:
        eig = np.linalg.eigvalsh(0.5 * (c + c.T))
    except np.linalg.LinAlgError:
        eig = np.array([])
        issues.append("eigen_failed")
    min_eig = float(np.min(eig)) if eig.size else float("nan")
    max_eig = float(np.max(eig)) if eig.size else float("nan")
    if eig.size and min_eig < -1e-10:
        issues.append("not_psd")
    cond = float(max_eig / max(min_eig, 1e-18)) if eig.size and max_eig > 0 else float("inf")
    if cond > 1e12:
        issues.append("ill_conditioned")
    if np.any(np.diag(c) <= 0):
        issues.append("nonpositive_variance")

    return {
        "name": "diagnose_covariance",
        "ok": len(issues) == 0,
        "issues": issues,
        "shape": [n, n],
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "condition_number": cond,
        "asymmetry": asym,
        "trace": float(np.trace(c)),
        "frobenius": float(np.linalg.norm(c, "fro")),
    }


def diagnose_weights(
    weights: Any,
    *,
    names: list[str] | None = None,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Assess weight vector numerical health."""
    if isinstance(weights, dict):
        keys = names or list(weights.keys())
        w = np.asarray([float(weights.get(k, 0.0)) for k in keys], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        keys = names
    issues: list[str] = []
    if w.size == 0:
        return {"name": "diagnose_weights", "ok": True, "issues": [], "n": 0}
    if not np.all(np.isfinite(w)):
        issues.append("non_finite")
    if np.any(w < -tol):
        issues.append("negative_weights")
    s = float(np.sum(w))
    if abs(s - 1.0) > 1e-4 and s > tol:
        issues.append("not_normalized")
    if s <= tol:
        issues.append("zero_mass")
    hhi = float(np.sum(np.square(np.maximum(w, 0.0))))
    return {
        "name": "diagnose_weights",
        "ok": len(issues) == 0,
        "issues": issues,
        "n": int(w.size),
        "sum": s,
        "min": float(np.min(w)),
        "max": float(np.max(w)),
        "hhi": hhi,
        "effective_n": float(1.0 / hhi) if hhi > 1e-18 else 0.0,
        "names": keys,
    }


def diagnose_allocation(
    cov: Any | None = None,
    weights: Any | None = None,
    *,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Combined diagnostics for an allocation problem."""
    out: dict[str, Any] = {"name": "diagnose_allocation"}
    if cov is not None:
        out["covariance"] = diagnose_covariance(cov)
    if weights is not None:
        out["weights"] = diagnose_weights(weights, names=names)
    parts = [out[k]["ok"] for k in ("covariance", "weights") if k in out]
    out["ok"] = all(parts) if parts else True
    return out
