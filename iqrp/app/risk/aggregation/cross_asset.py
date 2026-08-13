"""Cross-asset risk aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_weights
from iqrp.app.risk.portfolio.portfolio_risk import portfolio_volatility


def cross_asset_risk(
    asset_vols: Any,
    correlation: Any,
    weights: Any,
    *,
    asset_names: list[str] | None = None,
) -> dict[str, Any]:
    """Combine asset volatilities and correlation into portfolio cross-asset risk."""
    vols = np.asarray(asset_vols, dtype=np.float64).reshape(-1)
    corr = np.asarray(correlation, dtype=np.float64)
    n = vols.size
    if corr.shape != (n, n):
        raise ValueError("correlation must be n x n matching asset_vols")
    w = as_weights(weights, n=n)
    names = list(asset_names) if asset_names is not None else [f"asset_{i}" for i in range(n)]
    if len(names) < n:
        names = names + [f"asset_{i}" for i in range(len(names), n)]
    names = names[:n]

    # cov = D corr D
    d = np.diag(np.maximum(vols, 0.0))
    cov = d @ corr @ d
    # Symmetrize / PSD soft clip
    cov = 0.5 * (cov + cov.T)
    port_vol = portfolio_volatility(w, cov)

    # Marginal stand-alone vs diversified
    standalone = float(np.sum(np.abs(w) * vols))
    diversification_benefit = float(max(standalone - port_vol.value, 0.0))

    return {
        "name": "cross_asset_risk",
        "portfolio_volatility": port_vol.to_dict(),
        "standalone_volatility": RiskMeasure(
            name="standalone_volatility",
            value=standalone,
            unit="volatility",
            method="sum_abs_w_sigma",
        ).to_dict(),
        "diversification_benefit": diversification_benefit,
        "asset_vols": {names[i]: float(vols[i]) for i in range(n)},
        "weights": {names[i]: float(w[i]) for i in range(n)},
        "n_assets": n,
    }
