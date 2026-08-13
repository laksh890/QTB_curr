"""EWMA covariance estimator (call-through to risk market correlation)."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.market.correlation import ewma_covariance as risk_ewma_covariance

__VERSION__ = "1.0.0"


def ewma_covariance(
    returns: Any,
    *,
    lambda_: float = 0.94,
    version: str = __VERSION__,
) -> dict[str, Any]:
    """EWMA / RiskMetrics-style covariance.

    Delegates to ``iqrp.app.risk.market.correlation.ewma_covariance``.
    """
    out = risk_ewma_covariance(returns, lambda_=lambda_)
    return {
        **out,
        "name": "ewma_covariance",
        "method": "ewma",
        "version": version,
        "matrix": out["matrix"],
        "shape": out["shape"],
        "n_obs": out["n_obs"],
        "lambda": out.get("lambda", lambda_),
    }
