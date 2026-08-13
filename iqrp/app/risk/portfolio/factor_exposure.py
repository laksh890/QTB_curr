"""Simple factor exposure via multivariate OLS."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_returns


def factor_exposures(
    asset_returns: Any,
    factor_returns: Any,
    *,
    factor_names: list[str] | None = None,
) -> dict[str, Any]:
    """Estimate factor betas via OLS: r = F b + e.

    ``asset_returns``: (T,) series
    ``factor_returns``: (T, K) factor matrix
    Uses only contemporaneous / past aligned rows; no lead variables.
    """
    y = as_returns(asset_returns)
    F = np.asarray(factor_returns, dtype=np.float64)
    if F.ndim == 1:
        F = F.reshape(-1, 1)
    if F.ndim != 2:
        raise ValueError("factor_returns must be 1-D or 2-D")

    t = int(min(y.size, F.shape[0]))
    k = int(F.shape[1])
    names = list(factor_names) if factor_names is not None else [f"factor_{i}" for i in range(k)]
    if len(names) < k:
        names = names + [f"factor_{i}" for i in range(len(names), k)]
    names = names[:k]

    if t < k + 1 or k == 0:
        betas = np.zeros(k, dtype=np.float64)
        r2 = 0.0
        resid_vol = float(np.std(y[-t:], ddof=1)) if t > 1 else 0.0
    else:
        y_t = y[-t:]
        F_t = F[-t:]
        mask = np.isfinite(y_t) & np.all(np.isfinite(F_t), axis=1)
        y_c = y_t[mask]
        F_c = F_t[mask]
        if y_c.size < k + 1:
            betas = np.zeros(k, dtype=np.float64)
            r2 = 0.0
            resid_vol = 0.0
        else:
            # Add intercept
            X = np.column_stack([np.ones(y_c.size), F_c])
            try:
                coef, _, _, _ = np.linalg.lstsq(X, y_c, rcond=None)
            except np.linalg.LinAlgError:
                coef = np.zeros(k + 1, dtype=np.float64)
            betas = coef[1:]
            fitted = X @ coef
            ss_res = float(np.sum((y_c - fitted) ** 2))
            ss_tot = float(np.sum((y_c - np.mean(y_c)) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            resid = y_c - fitted
            resid_vol = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0

    measures = [
        RiskMeasure(
            name=f"exposure_{names[i]}",
            value=float(betas[i]) if i < betas.size else 0.0,
            unit="beta",
            method="ols",
            parameters={"factor": names[i]},
        ).to_dict()
        for i in range(k)
    ]

    return {
        "name": "factor_exposures",
        "betas": {names[i]: float(betas[i]) for i in range(k)},
        "measures": measures,
        "r_squared": r2,
        "residual_volatility": resid_vol,
        "n_obs": t,
        "n_factors": k,
    }
