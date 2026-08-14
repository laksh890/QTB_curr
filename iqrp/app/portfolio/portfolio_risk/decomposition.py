"""Factor risk contribution decomposition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from iqrp.app.risk.portfolio.portfolio_risk import portfolio_volatility


def factor_risk_decomposition(
    weights: Any,
    *,
    factor_loadings: Any,
    factor_cov: Any | None = None,
    idiosyncratic_var: Any | None = None,
    cov: Any | None = None,
    factor_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Decompose portfolio variance into factor + idiosyncratic contributions.

    Model: Σ = B F B' + D  (or use provided ``cov`` for total risk).
    Factor risk contribution uses Euler allocation on factor exposures.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    B = np.asarray(factor_loadings, dtype=np.float64)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    if B.shape[0] != w.size and B.shape[1] == w.size:
        B = B.T
    n, k = B.shape[0], B.shape[1]
    if w.size != n:
        ww = np.zeros(n, dtype=np.float64)
        ww[: min(n, w.size)] = w[: min(n, w.size)]
        w = ww

    if factor_cov is None:
        F = np.eye(k, dtype=np.float64)
    else:
        F = np.asarray(factor_cov, dtype=np.float64)
        if F.ndim == 1:
            F = np.diag(F)

    if idiosyncratic_var is None:
        D = np.zeros(n, dtype=np.float64)
    else:
        d = np.asarray(idiosyncratic_var, dtype=np.float64).reshape(-1)
        D = np.zeros(n, dtype=np.float64)
        D[: min(n, d.size)] = d[: min(n, d.size)]

    # Portfolio factor exposure f = B' w
    f_expo = B.T @ w
    # Factor variance contribution: f' F f
    factor_var = float(f_expo @ F @ f_expo)
    idio_var = float(w @ (D * w))
    model_var = factor_var + idio_var

    if cov is not None:
        c = np.asarray(cov, dtype=np.float64)
        total_var = float(w @ c @ w)
        port_vol = float(np.sqrt(max(total_var, 0.0)))
    else:
        total_var = model_var
        port_vol = float(np.sqrt(max(total_var, 0.0)))

    # Marginal factor risk: d(sigma)/d(f_j) ≈ (F f)_j / sigma
    Ff = F @ f_expo
    if port_vol <= 1e-12:
        mrc_f = np.zeros(k, dtype=np.float64)
        crc_f = np.zeros(k, dtype=np.float64)
    else:
        mrc_f = Ff / port_vol
        crc_f = f_expo * mrc_f

    names = list(factor_names) if factor_names is not None else [f"factor_{i}" for i in range(k)]
    if len(names) < k:
        names = names + [f"factor_{i}" for i in range(len(names), k)]

    crc_sum = float(np.sum(crc_f))
    pct = (crc_f / crc_sum).tolist() if abs(crc_sum) > 1e-12 else [0.0] * k

    return {
        "name": "factor_risk_decomposition",
        "portfolio_volatility": port_vol,
        "total_variance": total_var,
        "factor_variance": factor_var,
        "idiosyncratic_variance": idio_var,
        "factor_variance_share": float(factor_var / total_var) if total_var > 1e-18 else 0.0,
        "factor_exposures": {names[i]: float(f_expo[i]) for i in range(k)},
        "marginal_factor_risk": {names[i]: float(mrc_f[i]) for i in range(k)},
        "component_factor_risk": {names[i]: float(crc_f[i]) for i in range(k)},
        "percent_factor_risk": {names[i]: float(pct[i]) for i in range(k)},
        "n_factors": k,
        "n_assets": n,
    }


def risk_decomposition(
    weights: Any,
    cov: Any,
    *,
    factor_loadings: Any | None = None,
    factor_cov: Any | None = None,
    idiosyncratic_var: Any | None = None,
    factor_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Asset-level Euler decomposition plus optional factor decomposition."""
    from iqrp.app.portfolio.portfolio_risk.contribution import risk_contribution

    asset = risk_contribution(weights, cov)
    out: dict[str, Any] = {
        "name": "risk_decomposition",
        "asset": asset,
        "portfolio_volatility": float(portfolio_volatility(weights, cov).value),
    }
    if factor_loadings is not None:
        out["factor"] = factor_risk_decomposition(
            weights,
            factor_loadings=factor_loadings,
            factor_cov=factor_cov,
            idiosyncratic_var=idiosyncratic_var,
            cov=cov,
            factor_names=factor_names,
        )
    return out
