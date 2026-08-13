"""MCMC / VI convergence diagnostics (R-hat, ESS, autocorrelation)."""

from __future__ import annotations

from typing import Any

import numpy as np


def gelman_rubin(chains: list[np.ndarray]) -> float:
    """Gelman-Rubin R-hat for scalar parameter chains (list of 1-D arrays)."""
    if len(chains) < 2:
        return 1.0
    arrays = [np.asarray(c, dtype=np.float64).reshape(-1) for c in chains]
    n = min(len(a) for a in arrays)
    if n < 2:
        return 1.0
    arrays = [a[-n:] for a in arrays]
    len(arrays)
    chain_means = np.array([a.mean() for a in arrays])
    chain_vars = np.array([a.var(ddof=1) for a in arrays])
    w = float(np.mean(chain_vars))
    b = float(n * np.var(chain_means, ddof=1))
    var_hat = ((n - 1) / n) * w + b / n
    if w <= 1e-300:
        return 1.0
    return float(np.sqrt(var_hat / w))


def effective_sample_size(x: np.ndarray, *, max_lag: int | None = None) -> float:
    """ESS via positive autocorrelation sum."""
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    n = v.size
    if n < 3:
        return float(n)
    v = v - v.mean()
    var = float(np.dot(v, v) / n)
    if var <= 1e-300:
        return float(n)
    max_lag = int(max_lag if max_lag is not None else min(n // 2, 100))
    rho_sum = 0.0
    for lag in range(1, max_lag + 1):
        rho = float(np.dot(v[:-lag], v[lag:]) / (n * var))
        if rho < 0:
            break
        rho_sum += rho
    ess = n / max(1.0 + 2.0 * rho_sum, 1e-6)
    return float(np.clip(ess, 1.0, n))


def autocorrelation(x: np.ndarray, *, max_lag: int = 40) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    n = v.size
    if n == 0:
        return np.zeros(0)
    v = v - v.mean()
    var = float(np.dot(v, v) / max(n, 1))
    if var <= 1e-300:
        return np.ones(min(max_lag, n))
    out = np.empty(min(max_lag, n), dtype=np.float64)
    out[0] = 1.0
    for lag in range(1, out.size):
        out[lag] = float(np.dot(v[:-lag], v[lag:]) / (n * var))
    return out


def burn_in_suggestion(trace: np.ndarray, *, window: int = 20) -> int:
    """Heuristic burn-in: first index after which rolling mean stabilizes."""
    v = np.asarray(trace, dtype=np.float64).reshape(-1)
    if v.size < window * 2:
        return max(0, v.size // 4)
    roll = np.convolve(v, np.ones(window) / window, mode="valid")
    target = roll[-1]
    tol = 0.1 * (np.std(v) + 1e-6)
    for i, val in enumerate(roll):
        if abs(val - target) < tol:
            return int(i)
    return int(v.size // 5)


def convergence_report(
    traces: dict[str, list[np.ndarray]],
    *,
    acceptance_rate: float | None = None,
) -> dict[str, Any]:
    """
    ``traces`` maps parameter name -> list of per-chain 1-D arrays.
    """
    report: dict[str, Any] = {"parameters": {}, "acceptance_rate": acceptance_rate}
    r_hats = []
    ess_vals = []
    for name, chains in traces.items():
        flat = (
            np.concatenate([np.asarray(c).reshape(-1) for c in chains]) if chains else np.array([])
        )
        rhat = gelman_rubin(chains)
        ess = effective_sample_size(flat)
        acf = autocorrelation(flat, max_lag=min(30, max(flat.size // 2, 1)))
        burn = burn_in_suggestion(flat)
        report["parameters"][name] = {
            "r_hat": rhat,
            "ess": ess,
            "burn_in_suggestion": burn,
            "acf": acf.tolist(),
            "mean": float(np.mean(flat)) if flat.size else 0.0,
            "std": float(np.std(flat)) if flat.size else 0.0,
        }
        r_hats.append(rhat)
        ess_vals.append(ess)
    report["max_r_hat"] = float(max(r_hats)) if r_hats else 1.0
    report["min_ess"] = float(min(ess_vals)) if ess_vals else 0.0
    report["converged"] = bool(report["max_r_hat"] < 1.1 and report["min_ess"] > 10)
    return report
