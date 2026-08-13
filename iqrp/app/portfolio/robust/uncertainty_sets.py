"""Box and ellipsoidal uncertainty sets for expected returns / covariance."""

from __future__ import annotations

from typing import Any

import numpy as np


def box_uncertainty_mu(
    mu: Any,
    *,
    absolute: Any | None = None,
    relative: float = 0.0,
    kappa: float = 0.0,
    cov: Any | None = None,
) -> dict[str, Any]:
    """
    Box uncertainty: mu ± delta elementwise.

    delta_i = absolute_i if provided else relative*|mu_i| + kappa*sigma_i
    """
    m = np.asarray(mu, dtype=np.float64).reshape(-1)
    n = m.size
    if absolute is not None:
        delta = np.abs(np.asarray(absolute, dtype=np.float64).reshape(-1))
        if delta.size == 1:
            delta = np.full(n, float(delta[0]))
        if delta.size != n:
            raise ValueError("absolute delta size mismatch")
    else:
        delta = float(relative) * np.abs(m)
        if cov is not None and kappa != 0.0:
            c = np.asarray(cov, dtype=np.float64)
            sig = np.sqrt(np.maximum(np.diag(c), 0.0))
            delta = delta + float(kappa) * sig
        delta = np.maximum(delta, 0.0)
    return {
        "type": "box",
        "mu": m,
        "delta": delta,
        "lower": m - delta,
        "upper": m + delta,
        "n": n,
    }


def ellipsoidal_uncertainty_mu(
    mu: Any,
    cov: Any,
    *,
    rho: float = 1.0,
    tau: float = 1.0,
) -> dict[str, Any]:
    """
    Ellipsoidal set: { m | (m-mu)' (tau Σ)^{-1} (m-mu) <= rho^2 }.

    Worst-case return for weights w is w'μ - rho * sqrt(w' (tau Σ) w).
    """
    m = np.asarray(mu, dtype=np.float64).reshape(-1)
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] != m.size:
        raise ValueError("cov must be square matching mu")
    c = 0.5 * (c + c.T)
    scale = max(float(tau), 1e-12) * c
    return {
        "type": "ellipsoidal",
        "mu": m,
        "scale_cov": scale,
        "rho": float(rho),
        "tau": float(tau),
        "n": m.size,
    }


def box_uncertainty_cov(
    cov: Any,
    *,
    relative: float = 0.1,
    absolute: float = 0.0,
) -> dict[str, Any]:
    """Elementwise box around covariance entries (symmetric)."""
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square")
    c = 0.5 * (c + c.T)
    delta = float(relative) * np.abs(c) + float(absolute)
    delta = 0.5 * (delta + delta.T)
    return {
        "type": "box_cov",
        "cov": c,
        "delta": delta,
        "lower": c - delta,
        "upper": c + delta,
        "n": c.shape[0],
    }


def worst_case_mu(
    weights: Any,
    uncertainty: dict[str, Any],
) -> np.ndarray:
    """
    Return the worst-case mean vector for a long portfolio under the set.

    For box: mu_i - sign(w_i)*delta_i (adversary lowers return).
    For ellipsoid: returns an effective mu such that w'mu_wc = w'μ - rho||L'w||.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    utype = uncertainty.get("type")
    if utype == "box":
        m = np.asarray(uncertainty["mu"], dtype=np.float64)
        d = np.asarray(uncertainty["delta"], dtype=np.float64)
        return m - np.sign(w) * d
    if utype == "ellipsoidal":
        m = np.asarray(uncertainty["mu"], dtype=np.float64)
        scale = np.asarray(uncertainty["scale_cov"], dtype=np.float64)
        rho = float(uncertainty["rho"])
        # direction of worst-case perturbation in mean space: - scale^{1/2} u parallel to L'w
        # Effective linearization: mu_wc = mu - rho * (scale w) / sqrt(w'scale w)
        quad = float(w @ scale @ w)
        if quad <= 1e-18:
            return m
        return m - rho * (scale @ w) / np.sqrt(quad)
    raise ValueError(f"unknown uncertainty type: {utype}")


def worst_case_return(weights: Any, uncertainty: dict[str, Any]) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    utype = uncertainty.get("type")
    if utype == "box":
        m = np.asarray(uncertainty["mu"], dtype=np.float64)
        d = np.asarray(uncertainty["delta"], dtype=np.float64)
        return float(w @ m - np.abs(w) @ d)
    if utype == "ellipsoidal":
        m = np.asarray(uncertainty["mu"], dtype=np.float64)
        scale = np.asarray(uncertainty["scale_cov"], dtype=np.float64)
        rho = float(uncertainty["rho"])
        return float(w @ m - rho * np.sqrt(max(float(w @ scale @ w), 0.0)))
    raise ValueError(f"unknown uncertainty type: {utype}")
