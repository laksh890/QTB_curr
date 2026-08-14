"""Classic Black–Litterman posterior expected returns and covariance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__VERSION__ = "1.0.0"


def _as_1d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _as_2d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def equilibrium_returns(
    cov: Sequence[Sequence[float]] | np.ndarray,
    market_weights: Sequence[float] | np.ndarray,
    *,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """Implied equilibrium excess returns ``pi = delta * Sigma * w_mkt`` (+ rf)."""
    Sigma = _as_2d(cov, "cov")
    w = _as_1d(market_weights, "market_weights")
    n = Sigma.shape[0]
    if Sigma.shape[1] != n:
        raise ValueError("cov must be square")
    if w.size != n:
        raise ValueError(f"market_weights length {w.size} != cov dimension {n}")
    delta = float(risk_aversion)
    pi = delta * (Sigma @ w) + float(risk_free_rate)
    return pi


def black_litterman_posterior(
    cov: Sequence[Sequence[float]] | np.ndarray,
    *,
    market_weights: Sequence[float] | np.ndarray | None = None,
    market_caps: Sequence[float] | np.ndarray | None = None,
    equilibrium_mu: Sequence[float] | np.ndarray | None = None,
    P: Sequence[Sequence[float]] | np.ndarray | None = None,
    Q: Sequence[float] | np.ndarray | None = None,
    omega: Sequence[Sequence[float]] | np.ndarray | Sequence[float] | None = None,
    tau: float = 0.05,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
    names: Sequence[str] | None = None,
    version: str = __VERSION__,
) -> dict[str, Any]:
    """Classic Black–Litterman posterior mean and covariance.

    Equilibrium prior ``pi`` is taken from ``equilibrium_mu`` when provided,
    otherwise implied from ``market_weights`` or capitalization-normalized
    ``market_caps`` via ``pi = delta * Sigma * w``.

    Views: ``P`` (K x N), ``Q`` (K,), and ``omega`` (K x K or diagonal of
    length K). When ``omega`` is omitted, uses ``diag(P (tau Sigma) P')``.

    Posterior::

        mu_bl = inv(inv(tau Sigma) + P' inv(Omega) P)
                @ (inv(tau Sigma) pi + P' inv(Omega) Q)

        Sigma_bl = Sigma + inv(inv(tau Sigma) + P' inv(Omega) P)
    """
    Sigma = _as_2d(cov, "cov")
    n = Sigma.shape[0]
    if Sigma.shape[1] != n:
        raise ValueError("cov must be square")
    Sigma = 0.5 * (Sigma + Sigma.T)

    if equilibrium_mu is not None:
        pi = _as_1d(equilibrium_mu, "equilibrium_mu")
        if pi.size != n:
            raise ValueError(f"equilibrium_mu length {pi.size} != N={n}")
        eq_method = "provided"
        w_mkt = None
    else:
        if market_weights is not None:
            w_mkt = _as_1d(market_weights, "market_weights")
            eq_method = "market_weights"
        elif market_caps is not None:
            caps = _as_1d(market_caps, "market_caps")
            if caps.size != n:
                raise ValueError(f"market_caps length {caps.size} != N={n}")
            total = float(np.sum(np.maximum(caps, 0.0)))
            if total <= 0.0:
                w_mkt = np.full(n, 1.0 / n, dtype=np.float64)
            else:
                w_mkt = np.maximum(caps, 0.0) / total
            eq_method = "market_caps"
        else:
            w_mkt = np.full(n, 1.0 / max(n, 1), dtype=np.float64)
            eq_method = "equal_weight"
        if w_mkt.size != n:
            raise ValueError(f"market_weights length {w_mkt.size} != N={n}")
        pi = equilibrium_returns(
            Sigma,
            w_mkt,
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
        )

    tau = float(max(tau, 1e-12))
    tau_sigma = tau * Sigma
    tau_sigma_inv = np.linalg.pinv(tau_sigma + 1e-12 * np.eye(n))

    if P is None or Q is None:
        # No views: posterior mean is equilibrium; posterior cov = Sigma + tau Sigma
        mu_post = pi.copy()
        cov_post = Sigma + tau_sigma
        k = 0
        view_method = "none"
        omega_out: list[Any] = []
        P_out: list[Any] = []
        Q_out: list[Any] = []
    else:
        P_mat = _as_2d(P, "P")
        Q_vec = _as_1d(Q, "Q")
        k = P_mat.shape[0]
        if P_mat.shape[1] != n:
            raise ValueError(f"P columns {P_mat.shape[1]} != N={n}")
        if Q_vec.size != k:
            raise ValueError(f"Q length {Q_vec.size} != K={k}")

        if omega is None:
            Omega = np.diag(np.maximum(np.diag(P_mat @ tau_sigma @ P_mat.T), 1e-12))
            omega_method = "proportional_to_P_tauSigma_P"
        else:
            om = np.asarray(omega, dtype=np.float64)
            if om.ndim == 1:
                if om.size != k:
                    raise ValueError(f"omega diagonal length {om.size} != K={k}")
                Omega = np.diag(np.maximum(om, 1e-12))
                omega_method = "diagonal_provided"
            else:
                Omega = _as_2d(om, "omega")
                if Omega.shape != (k, k):
                    raise ValueError(f"omega shape {Omega.shape} != ({k}, {k})")
                Omega = 0.5 * (Omega + Omega.T)
                Omega = Omega + 1e-12 * np.eye(k)
                omega_method = "matrix_provided"

        Omega_inv = np.linalg.pinv(Omega)
        # Precision form
        precision = tau_sigma_inv + P_mat.T @ Omega_inv @ P_mat
        cov_view = np.linalg.pinv(precision)
        rhs = tau_sigma_inv @ pi + P_mat.T @ Omega_inv @ Q_vec
        mu_post = cov_view @ rhs
        cov_post = Sigma + cov_view
        view_method = omega_method
        omega_out = Omega.tolist()
        P_out = P_mat.tolist()
        Q_out = Q_vec.tolist()

    cov_post = 0.5 * (cov_post + cov_post.T)
    cov_post = np.nan_to_num(cov_post, nan=0.0, posinf=0.0, neginf=0.0)
    mu_post = np.nan_to_num(mu_post, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "name": "black_litterman_posterior",
        "method": "black_litterman",
        "mu": mu_post.tolist(),
        "vector": mu_post.tolist(),
        "matrix": cov_post.tolist(),
        "posterior_cov": cov_post.tolist(),
        "shape": [n],
        "cov_shape": list(cov_post.shape),
        "n_obs": int(n),
        "n_views": int(k),
        "tau": tau,
        "risk_aversion": float(risk_aversion),
        "risk_free_rate": float(risk_free_rate),
        "equilibrium_mu": pi.tolist(),
        "equilibrium_method": eq_method,
        "market_weights": w_mkt.tolist() if w_mkt is not None else None,
        "P": P_out,
        "Q": Q_out,
        "omega": omega_out,
        "view_method": view_method,
        "names": list(names) if names is not None else None,
        "version": version,
    }
