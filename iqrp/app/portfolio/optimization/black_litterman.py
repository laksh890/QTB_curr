"""Black–Litterman posterior mean-variance optimization."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.portfolio.optimization.mean_variance import optimize_mean_variance
from iqrp.app.portfolio.optimization.projection import (
    as_cov,
    as_vector,
    failed_result,
    make_result,
)


def _local_black_litterman_posterior(
    cov: np.ndarray,
    *,
    market_weights: np.ndarray | None = None,
    risk_aversion: float = 1.0,
    P: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    tau: float = 0.05,
    equilibrium_returns: np.ndarray | None = None,
) -> dict[str, Any]:
    """Classic BL posterior (used if expected_returns module is unavailable)."""
    n = cov.shape[0]
    pi = (
        as_vector(equilibrium_returns, n)
        if equilibrium_returns is not None
        else float(risk_aversion) * (cov @ (as_vector(market_weights, n) if market_weights is not None else np.full(n, 1.0 / n)))
    )
    if P is None or Q is None:
        # no views → posterior = prior
        return {
            "mu": pi,
            "posterior_mu": pi,
            "posterior_cov": cov,
            "method": "black_litterman_local_no_views",
            "tau": float(tau),
        }
    P_m = np.asarray(P, dtype=np.float64)
    Q_v = as_vector(Q)
    if P_m.ndim == 1:
        P_m = P_m.reshape(1, -1)
    if P_m.shape[1] != n:
        raise ValueError("P must have n columns")
    if Q_v.size != P_m.shape[0]:
        raise ValueError("Q length must match P rows")
    if omega is None:
        # He–Litterman diagonal proportional to P (tau Σ) P'
        omega_m = np.diag(np.maximum(np.diag(P_m @ (tau * cov) @ P_m.T), 1e-12))
    else:
        omega_m = np.asarray(omega, dtype=np.float64)
        if omega_m.ndim == 1:
            omega_m = np.diag(omega_m)
    tau_sig = tau * cov
    # posterior mean: pi + tauΣ P' (P tauΣ P' + Ω)^{-1} (Q - P pi)
    mid = P_m @ tau_sig @ P_m.T + omega_m
    adj = tau_sig @ P_m.T @ np.linalg.solve(mid, (Q_v - P_m @ pi))
    mu_post = pi + adj
    # posterior cov of returns (common BL form)
    cov_post = cov + np.linalg.inv(np.linalg.inv(tau_sig) + P_m.T @ np.linalg.solve(omega_m, P_m))
    cov_post = 0.5 * (cov_post + cov_post.T)
    return {
        "mu": mu_post,
        "posterior_mu": mu_post,
        "posterior_cov": cov_post,
        "equilibrium_returns": pi,
        "method": "black_litterman_local",
        "tau": float(tau),
    }


def _call_bl_posterior(
    cov: np.ndarray,
    *,
    market_weights: Any,
    risk_aversion: float,
    P: Any,
    Q: Any,
    omega: Any,
    tau: float,
    equilibrium_returns: Any,
) -> dict[str, Any]:
    try:
        from iqrp.app.portfolio.expected_returns.black_litterman import (  # type: ignore
            black_litterman_posterior,
        )

        try:
            return black_litterman_posterior(
                cov,
                market_weights=market_weights,
                risk_aversion=risk_aversion,
                P=P,
                Q=Q,
                omega=omega,
                tau=tau,
                equilibrium_returns=equilibrium_returns,
            )
        except TypeError:
            # alternate keyword styles
            kwargs: dict[str, Any] = {"tau": tau}
            if market_weights is not None:
                kwargs["market_weights"] = market_weights
            if P is not None:
                kwargs["P"] = P
            if Q is not None:
                kwargs["Q"] = Q
            if omega is not None:
                kwargs["omega"] = omega
            if equilibrium_returns is not None:
                kwargs["pi"] = equilibrium_returns
            kwargs["delta"] = risk_aversion
            return black_litterman_posterior(cov, **kwargs)
    except Exception:
        return _local_black_litterman_posterior(
            cov,
            market_weights=None if market_weights is None else as_vector(market_weights, cov.shape[0]),
            risk_aversion=risk_aversion,
            P=None if P is None else np.asarray(P, dtype=np.float64),
            Q=None if Q is None else as_vector(Q),
            omega=None if omega is None else np.asarray(omega, dtype=np.float64),
            tau=tau,
            equilibrium_returns=None if equilibrium_returns is None else as_vector(equilibrium_returns, cov.shape[0]),
        )


def optimize_black_litterman(
    mu: Any = None,
    cov: Any = None,
    *,
    current_weights: Any = None,
    constraints: Any = None,
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_aversion: float = 1.0,
    market_weights: Any = None,
    P: Any = None,
    Q: Any = None,
    omega: Any = None,
    tau: float = 0.05,
    equilibrium_returns: Any = None,
    use_posterior_cov: bool = False,
    min_weight: float | None = None,
    max_gross: float | None = None,
    budget: float = 1.0,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build Black–Litterman posterior expected returns, then run mean-variance.

    Prefers ``iqrp.app.portfolio.expected_returns.black_litterman.black_litterman_posterior``.
    """
    name = "black_litterman"
    method = "bl_posterior_mv"
    try:
        if cov is None:
            raise ValueError("cov is required")
        c = as_cov(cov)
        n = c.shape[0]
        # If caller passed mu without views, treat as equilibrium prior
        eq = equilibrium_returns if equilibrium_returns is not None else mu

        post = _call_bl_posterior(
            c,
            market_weights=market_weights,
            risk_aversion=risk_aversion,
            P=P,
            Q=Q,
            omega=omega,
            tau=tau,
            equilibrium_returns=eq,
        )
        mu_post = post.get("posterior_mu", post.get("mu"))
        if mu_post is None:
            raise ValueError("BL posterior missing mu")
        mu_v = as_vector(mu_post, n)
        cov_use = as_cov(post["posterior_cov"], n) if use_posterior_cov and post.get("posterior_cov") is not None else c

        mv = optimize_mean_variance(
            mu=mu_v,
            cov=cov_use,
            current_weights=current_weights,
            constraints=constraints,
            long_only=long_only,
            max_weight=max_weight,
            risk_aversion=risk_aversion,
            min_weight=min_weight,
            max_gross=max_gross,
            budget=budget,
            names=names,
        )
        out = dict(mv)
        out["name"] = name
        out["method"] = f"{method}+{mv.get('method')}"
        diag = dict(out.get("diagnostics") or {})
        diag["bl_backend"] = post.get("method", "black_litterman")
        diag["tau"] = float(tau)
        diag["posterior_mu"] = mu_v.tolist()
        diag["use_posterior_cov"] = bool(use_posterior_cov)
        out["diagnostics"] = diag
        if not mv.get("success"):
            return out
        return make_result(
            name,
            out["weights"],
            success=True,
            status=out.get("status", "optimal"),
            method=out["method"],
            failure_reason=out.get("failure_reason"),
            conflicting_constraints=out.get("conflicting_constraints"),
            diagnostics=diag,
            objective_value=out.get("objective_value"),
        )
    except Exception as exc:
        n = 0
        try:
            n = int(np.asarray(cov).shape[0]) if cov is not None else 0
        except Exception:
            n = 0
        return failed_result(name, n, method=method, reason=str(exc))
