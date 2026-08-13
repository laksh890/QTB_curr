"""Parameter uncertainty robustification for mu / cov."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.portfolio.optimization.mean_variance import optimize_mean_variance
from iqrp.app.portfolio.optimization.projection import (
    as_cov,
    as_vector,
    failed_result,
    make_result,
    stabilize_mu,
)
from iqrp.app.portfolio.robust.distributional_robust import optimize_distributional_robust
from iqrp.app.portfolio.robust.uncertainty_sets import box_uncertainty_mu, ellipsoidal_uncertainty_mu


def shrink_covariance(cov: Any, *, intensity: float = 0.2) -> np.ndarray:
    """Linear shrinkage toward scaled identity."""
    c = as_cov(cov)
    n = c.shape[0]
    mu_var = float(np.trace(c) / n)
    alpha = float(np.clip(intensity, 0.0, 1.0))
    return (1.0 - alpha) * c + alpha * mu_var * np.eye(n)


def mu_standard_errors(
    returns: Any | None = None,
    *,
    cov: Any | None = None,
    n_obs: int | None = None,
) -> np.ndarray:
    """Diagonal SE for means: sigma / sqrt(T)."""
    if returns is not None:
        r = np.asarray(returns, dtype=np.float64)
        if r.ndim != 2:
            raise ValueError("returns must be (T, N)")
        t = r.shape[0]
        sig = np.std(r, axis=0, ddof=1)
        return sig / max(np.sqrt(t), 1.0)
    if cov is None:
        raise ValueError("returns or cov required")
    c = as_cov(cov)
    t = int(n_obs or c.shape[0] * 20)
    return np.sqrt(np.maximum(np.diag(c), 0.0)) / max(np.sqrt(t), 1.0)


def optimize_parameter_uncertainty(
    mu: Any = None,
    cov: Any = None,
    *,
    returns: Any = None,
    current_weights: Any = None,
    constraints: Any = None,
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_aversion: float = 1.0,
    z_score: float = 1.0,
    cov_shrink: float = 0.2,
    set_type: str = "box",
    rho: float = 1.0,
    tau: float = 0.05,
    min_weight: float | None = None,
    max_gross: float | None = None,
    budget: float = 1.0,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Robustify estimation error:

    - shrink covariance
    - build mu uncertainty from mean SEs (box) or ellipsoidal set
    - optimize worst-case MV
    """
    name = "parameter_uncertainty"
    method = "se_uncertainty_mv"
    try:
        if cov is None and returns is None:
            raise ValueError("cov or returns required")
        if returns is not None:
            r = np.asarray(returns, dtype=np.float64)
            n = r.shape[1]
            c_raw = as_cov(np.cov(r, rowvar=False)) if cov is None else as_cov(cov, n)
            m = stabilize_mu(np.mean(r, axis=0) if mu is None else as_vector(mu, n))
            se = mu_standard_errors(r)
        else:
            c_raw = as_cov(cov)
            n = c_raw.shape[0]
            m = np.zeros(n) if mu is None else stabilize_mu(as_vector(mu, n))
            se = mu_standard_errors(cov=c_raw)

        c = shrink_covariance(c_raw, intensity=cov_shrink)
        kind = str(set_type).lower()
        if kind == "ellipsoidal":
            uset = ellipsoidal_uncertainty_mu(m, c, rho=rho, tau=tau)
        else:
            uset = box_uncertainty_mu(m, absolute=float(z_score) * se)

        res = optimize_distributional_robust(
            mu=m,
            cov=c,
            current_weights=current_weights,
            constraints=constraints,
            long_only=long_only,
            max_weight=max_weight,
            risk_aversion=risk_aversion,
            uncertainty=uset,
            min_weight=min_weight,
            max_gross=max_gross,
            budget=budget,
            names=names,
        )
        out = dict(res)
        out["name"] = name
        diag = dict(out.get("diagnostics") or {})
        diag["set_type"] = uset.get("type")
        diag["mean_se"] = se.tolist()
        diag["cov_shrink"] = float(cov_shrink)
        diag["z_score"] = float(z_score)
        out["diagnostics"] = diag
        out["method"] = f"{method}+{res.get('method')}"
        return make_result(
            name,
            out["weights"],
            success=bool(out.get("success")),
            status=str(out.get("status", "failed")),
            method=out["method"],
            failure_reason=out.get("failure_reason"),
            conflicting_constraints=out.get("conflicting_constraints"),
            diagnostics=diag,
            objective_value=out.get("objective_value"),
        )
    except Exception as exc:
        n = 0
        try:
            if returns is not None:
                n = int(np.asarray(returns).shape[1])
            elif cov is not None:
                n = int(np.asarray(cov).shape[0])
        except Exception:
            n = 0
        return failed_result(name, n, method=method, reason=str(exc))


def optimize_robust_mean_variance(
    mu: Any = None,
    cov: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience: parameter-uncertainty robust MV with sensible defaults."""
    return optimize_parameter_uncertainty(mu=mu, cov=cov, **kwargs)
