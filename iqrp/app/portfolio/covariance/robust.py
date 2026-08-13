"""Robust covariance estimators (winsorized sample / minimum covariance determinant style)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.risk.market.correlation import covariance_matrix

__VERSION__ = "1.0.0"


def _as_matrix(returns: Any) -> np.ndarray:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("returns must be 1-D or 2-D (T x N)")
    return arr


def _winsorize(x: np.ndarray, limits: tuple[float, float] = (0.05, 0.05)) -> np.ndarray:
    lo_q, hi_q = float(limits[0]), float(limits[1])
    out = x.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        finite = np.isfinite(col)
        if not np.any(finite):
            continue
        lo = float(np.quantile(col[finite], lo_q))
        hi = float(np.quantile(col[finite], 1.0 - hi_q))
        out[:, j] = np.clip(col, lo, hi)
    return out


def _mcd_subset_covariance(
    x: np.ndarray,
    *,
    h_fraction: float = 0.75,
    n_trials: int = 32,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Simple deterministic MCD-style estimator via random halfspace subsets.

    Selects the subset of size ``h`` with the smallest determinant covariance.
    """
    t, n = x.shape
    if t < 2 or n == 0:
        return np.zeros((n, n), dtype=np.float64), 0

    mask = np.all(np.isfinite(x), axis=1)
    clean = x[mask]
    t_eff = int(clean.shape[0])
    if t_eff < 2:
        return np.zeros((n, n), dtype=np.float64), 0

    h = max(n + 1, int(np.ceil(float(h_fraction) * t_eff)))
    h = min(h, t_eff)
    rng = np.random.default_rng(int(seed))

    best_cov: np.ndarray | None = None
    best_det = np.inf
    trials = max(1, min(int(n_trials), 200))

    for _ in range(trials):
        idx = rng.choice(t_eff, size=h, replace=False)
        subset = clean[idx]
        cov = np.cov(subset, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=np.float64)
        # Stabilize and score by determinant
        cov = cov + 1e-12 * np.eye(n)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            continue
        det = float(np.exp(logdet))
        if det < best_det:
            best_det = det
            best_cov = cov

    if best_cov is None:
        cov = np.cov(clean, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=np.float64)
        best_cov = cov

    best_cov = np.nan_to_num(best_cov, nan=0.0, posinf=0.0, neginf=0.0)
    best_cov = 0.5 * (best_cov + best_cov.T)
    return best_cov, t_eff


def robust_covariance(
    returns: Any,
    *,
    method: Literal["winsorize", "mcd", "winsorize_mcd"] = "winsorize",
    winsor_limits: tuple[float, float] = (0.05, 0.05),
    h_fraction: float = 0.75,
    n_trials: int = 32,
    seed: int = 42,
    version: str = __VERSION__,
) -> dict[str, Any]:
    """Robust covariance via winsorization and/or a simple MCD-style subset search."""
    x = _as_matrix(returns)
    t, n = x.shape

    if t < 2 or n == 0:
        base = covariance_matrix(returns)
        return {
            **base,
            "name": "robust_covariance",
            "method": method,
            "version": version,
        }

    x_work = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    used_method = method

    if method in ("winsorize", "winsorize_mcd"):
        x_work = _winsorize(x_work, limits=winsor_limits)

    if method in ("mcd", "winsorize_mcd"):
        cov, n_obs = _mcd_subset_covariance(
            x_work,
            h_fraction=h_fraction,
            n_trials=n_trials,
            seed=seed,
        )
    else:
        out = covariance_matrix(x_work)
        cov = np.asarray(out["matrix"], dtype=np.float64)
        n_obs = int(out["n_obs"])
        used_method = "winsorize_sample"

    return {
        "name": "robust_covariance",
        "method": used_method,
        "matrix": cov.tolist(),
        "shape": list(cov.shape),
        "n_obs": int(n_obs),
        "winsor_limits": [float(winsor_limits[0]), float(winsor_limits[1])],
        "h_fraction": float(h_fraction),
        "n_trials": int(n_trials),
        "seed": int(seed),
        "version": version,
    }
