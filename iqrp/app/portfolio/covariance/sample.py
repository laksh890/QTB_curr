"""Sample covariance estimator (call-through to risk market correlation)."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.market.correlation import covariance_matrix

__VERSION__ = "1.0.0"


def sample_covariance(
    returns: Any,
    *,
    window: int | None = None,
    version: str = __VERSION__,
) -> dict[str, Any]:
    """Sample covariance matrix of asset returns.

    Delegates to ``iqrp.app.risk.market.correlation.covariance_matrix``.
    """
    out = covariance_matrix(returns, window=window)
    return {
        **out,
        "name": "sample_covariance",
        "method": "sample",
        "version": version,
        "matrix": out["matrix"],
        "shape": out["shape"],
        "n_obs": out["n_obs"],
    }
