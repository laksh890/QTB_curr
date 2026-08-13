"""MLE utilities for volatility models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from iqrp.app.forecasting.volatility.base.distributions import logpdf


@dataclass(slots=True)
class MLEResult:
    params: np.ndarray
    param_names: list[str]
    loglik: float
    aic: float
    bic: float
    success: bool
    variance: np.ndarray
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params.tolist(),
            "param_names": list(self.param_names),
            "loglik": self.loglik,
            "aic": self.aic,
            "bic": self.bic,
            "success": self.success,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


def gaussian_nll_from_variance(
    returns: np.ndarray,
    variance: np.ndarray,
    *,
    dist: str = "gaussian",
    dist_kwargs: dict[str, float] | None = None,
) -> float:
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    v = np.clip(np.asarray(variance, dtype=np.float64).reshape(-1), 1e-12, None)
    n = min(r.size, v.size)
    r, v = r[:n], v[:n]
    z = r / np.sqrt(v)
    kw = dist_kwargs or {}
    ll = -0.5 * np.log(v) + logpdf(z, name=dist, **kw)
    return float(-np.sum(ll))


def estimate(
    returns: np.ndarray,
    variance_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    *,
    param_names: list[str],
    dist: str = "gaussian",
    dist_kwargs: dict[str, float] | None = None,
    method: str = "L-BFGS-B",
    maxiter: int = 500,
    n_restarts: int = 1,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> MLEResult:
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    best: MLEResult | None = None
    starts = [np.asarray(x0, dtype=np.float64)]
    rng = np.random.default_rng(0)
    for _ in range(max(int(n_restarts) - 1, 0)):
        jitter = starts[0] * (1 + 0.1 * rng.normal(size=starts[0].size))
        starts.append(np.clip(jitter, [b[0] for b in bounds], [b[1] for b in bounds]))

    def objective(theta: np.ndarray) -> float:
        p = transform(theta) if transform is not None else theta
        try:
            var = variance_fn(p)
        except Exception:  # noqa: BLE001
            return 1e20
        if not np.all(np.isfinite(var)):
            return 1e20
        val = gaussian_nll_from_variance(r, var, dist=dist, dist_kwargs=dist_kwargs)
        return val if np.isfinite(val) else 1e20

    for start in starts:
        try:
            if method == "robust":
                res = minimize(objective, start, method="Nelder-Mead", options={"maxiter": maxiter})
                res2 = minimize(
                    objective,
                    res.x,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": maxiter},
                )
                res = res2 if res2.fun <= res.fun else res
            elif method == "Nelder-Mead":
                res = minimize(objective, start, method="Nelder-Mead", options={"maxiter": maxiter})
            else:
                res = minimize(
                    objective,
                    start,
                    method=method,
                    bounds=bounds,
                    options={"maxiter": maxiter},
                )
        except Exception:  # noqa: BLE001
            continue
        theta = np.asarray(res.x, dtype=np.float64)
        params = transform(theta) if transform is not None else theta
        try:
            var = variance_fn(params)
        except Exception:  # noqa: BLE001
            continue
        if not np.all(np.isfinite(var)):
            continue
        nll = float(res.fun)
        ll = -nll
        k = params.size
        n = r.size
        cand = MLEResult(
            params=params,
            param_names=list(param_names),
            loglik=ll,
            aic=-2 * ll + 2 * k,
            bic=-2 * ll + k * np.log(max(n, 1)),
            success=bool(getattr(res, "success", False)),
            variance=np.asarray(var, dtype=np.float64),
            message=str(getattr(res, "message", "")),
        )
        if best is None or cand.loglik > best.loglik:
            best = cand
    if best is None:
        x0_arr = np.asarray(x0, dtype=np.float64)
        try:
            var = variance_fn(x0_arr)
        except Exception:  # noqa: BLE001
            var = np.full(r.size, max(float(np.mean(r**2)), 1e-6))
        if not np.all(np.isfinite(var)):
            var = np.full(r.size, max(float(np.mean(r**2)), 1e-6))
        nll = gaussian_nll_from_variance(r, var, dist=dist, dist_kwargs=dist_kwargs)
        best = MLEResult(
            params=x0_arr,
            param_names=list(param_names),
            loglik=-nll,
            aic=2 * nll + 2 * len(param_names),
            bic=2 * nll + len(param_names) * np.log(max(r.size, 1)),
            success=False,
            variance=np.asarray(var, dtype=np.float64),
            message="fallback",
        )
    return best
