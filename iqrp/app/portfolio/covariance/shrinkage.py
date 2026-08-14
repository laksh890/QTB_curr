"""Shrinkage covariance estimators (Ledoit–Wolf style + risk call-through)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.risk.market.correlation import (
    covariance_matrix,
    shrinkage_covariance as risk_shrinkage,
)

__VERSION__ = "1.0.0"


def _as_matrix(returns: Any) -> np.ndarray:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("returns must be 1-D or 2-D (T x N)")
    return arr


def ledoit_wolf_covariance(
    returns: Any,
    *,
    version: str = __VERSION__,
) -> dict[str, Any]:
    """Ledoit–Wolf shrinkage intensity toward scaled identity (analytic estimate).

    Uses the constant-correlation / identity-target intensity heuristic on the
    sample covariance of finite rows. Falls back to risk shrinkage when T < 2.
    """
    x = _as_matrix(returns)
    t, n = x.shape
    if t < 2 or n == 0:
        base = risk_shrinkage(returns)
        return {
            **base,
            "name": "ledoit_wolf_covariance",
            "method": "ledoit_wolf",
            "version": version,
        }

    mask = np.all(np.isfinite(x), axis=1)
    clean = x[mask]
    t_eff = int(clean.shape[0])
    if t_eff < 2:
        base = risk_shrinkage(returns)
        return {
            **base,
            "name": "ledoit_wolf_covariance",
            "method": "ledoit_wolf",
            "version": version,
        }

    sample = np.cov(clean, rowvar=False)
    if sample.ndim == 0:
        sample = np.array([[float(sample)]], dtype=np.float64)
    sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)

    mu = float(np.trace(sample) / n)
    target = mu * np.eye(n, dtype=np.float64)

    # Analytic LW intensity for identity target (simplified Ledoit–Wolf 2004)
    x_dm = clean - clean.mean(axis=0, keepdims=True)
    # Sum of squared deviations of sample moments
    y = x_dm**2
    phi_mat = (y.T @ y) / t_eff - sample**2
    phi = float(np.sum(phi_mat))
    gamma = float(np.linalg.norm(sample - target, "fro") ** 2)
    kappa = phi / gamma if gamma > 1e-18 else 1.0
    intensity = float(np.clip(kappa / t_eff, 0.0, 1.0))

    shrunk = (1.0 - intensity) * sample + intensity * target
    return {
        "name": "ledoit_wolf_covariance",
        "method": "ledoit_wolf",
        "matrix": shrunk.tolist(),
        "shape": list(shrunk.shape),
        "n_obs": t_eff,
        "intensity": intensity,
        "target_trace_mean": mu,
        "version": version,
    }


def shrinkage_covariance(
    returns: Any,
    *,
    intensity: float | None = None,
    method: Literal["risk", "ledoit_wolf"] = "risk",
    version: str = __VERSION__,
) -> dict[str, Any]:
    """Shrinkage covariance.

    ``method='risk'`` delegates to ``iqrp.app.risk.market.correlation.shrinkage_covariance``.
    ``method='ledoit_wolf'`` uses the analytic Ledoit–Wolf intensity estimate
    (``intensity`` overrides when provided).
    """
    if method == "ledoit_wolf" and intensity is None:
        return ledoit_wolf_covariance(returns, version=version)

    if method == "ledoit_wolf" and intensity is not None:
        sample = covariance_matrix(returns)
        cov = np.asarray(sample["matrix"], dtype=np.float64)
        n = cov.shape[0]
        if n == 0:
            return {
                **sample,
                "name": "shrinkage_covariance",
                "method": "ledoit_wolf",
                "intensity": 0.0,
                "version": version,
            }
        mu = float(np.trace(cov) / n)
        target = mu * np.eye(n)
        alpha = float(np.clip(intensity, 0.0, 1.0))
        shrunk = (1.0 - alpha) * cov + alpha * target
        return {
            "name": "shrinkage_covariance",
            "method": "ledoit_wolf",
            "matrix": shrunk.tolist(),
            "shape": list(shrunk.shape),
            "n_obs": sample["n_obs"],
            "intensity": alpha,
            "version": version,
        }

    out = risk_shrinkage(returns, intensity=intensity)
    return {
        **out,
        "name": "shrinkage_covariance",
        "method": out.get("method", "shrinkage_identity"),
        "version": version,
        "matrix": out["matrix"],
        "shape": out["shape"],
        "n_obs": out["n_obs"],
    }
