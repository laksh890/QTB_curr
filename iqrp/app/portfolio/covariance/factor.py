"""Factor-model covariance: B F B' + D."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.market.correlation import covariance_matrix

__VERSION__ = "1.0.0"


def _as_2d(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D")
    return arr


def factor_covariance(
    *,
    factor_loadings: Any,
    factor_returns: Any | None = None,
    factor_cov: Any | None = None,
    residual_vars: Any | None = None,
    asset_returns: Any | None = None,
    version: str = __VERSION__,
) -> dict[str, Any]:
    """Build asset covariance as ``B F B' + D``.

    Parameters
    ----------
    factor_loadings:
        Loadings matrix ``B`` with shape (N assets, K factors).
    factor_returns:
        Optional factor return matrix (T x K). Used to estimate ``F`` when
        ``factor_cov`` is not provided.
    factor_cov:
        Optional factor covariance ``F`` (K x K).
    residual_vars:
        Optional residual variances length N (diagonal of ``D``). When omitted
        and ``asset_returns`` is given, residuals are estimated from a linear
        projection; otherwise ``D`` is a small ridge diagonal.
    asset_returns:
        Optional asset returns (T x N) for residual variance estimation.
    """
    B = _as_2d(factor_loadings, name="factor_loadings")
    n, k = B.shape

    if factor_cov is not None:
        F = np.asarray(factor_cov, dtype=np.float64)
        if F.shape != (k, k):
            raise ValueError(f"factor_cov shape {F.shape} incompatible with loadings K={k}")
        n_obs = 0
        factor_method = "provided"
    elif factor_returns is not None:
        fr = _as_2d(factor_returns, name="factor_returns")
        if fr.shape[1] != k:
            raise ValueError(f"factor_returns columns {fr.shape[1]} != loadings K={k}")
        cov_out = covariance_matrix(fr)
        F = np.asarray(cov_out["matrix"], dtype=np.float64)
        if F.size == 0:
            F = np.eye(k, dtype=np.float64)
        n_obs = int(cov_out["n_obs"])
        factor_method = "sample_factor_returns"
    else:
        F = np.eye(k, dtype=np.float64)
        n_obs = 0
        factor_method = "identity_prior"

    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    if residual_vars is not None:
        d = np.asarray(residual_vars, dtype=np.float64).reshape(-1)
        if d.size != n:
            raise ValueError(f"residual_vars length {d.size} != N={n}")
        d = np.maximum(d, 0.0)
        residual_method = "provided"
    elif asset_returns is not None and factor_returns is not None:
        a = _as_2d(asset_returns, name="asset_returns")
        fr = _as_2d(factor_returns, name="factor_returns")
        t = min(a.shape[0], fr.shape[0])
        a = a[-t:]
        fr = fr[-t:]
        if a.shape[1] != n:
            raise ValueError(f"asset_returns columns {a.shape[1]} != N={n}")
        # OLS residuals per asset: r = f @ b + e  (using provided loadings)
        fitted = fr @ B.T
        resid = a - fitted
        if resid.shape[0] > 1:
            d = np.var(resid, axis=0, ddof=1)
        else:
            d = np.var(resid, axis=0, ddof=0)
        d = np.maximum(np.nan_to_num(d, nan=0.0), 1e-12)
        n_obs = max(n_obs, int(t))
        residual_method = "ols_residuals"
    else:
        # Small ridge so the matrix is usable without inventing structure
        diag_bfb = np.maximum(np.diag(B @ F @ B.T), 0.0)
        scale = float(np.mean(diag_bfb)) if n else 1e-8
        d = np.full(n, max(scale * 0.05, 1e-8), dtype=np.float64)
        residual_method = "ridge_diagonal"

    D = np.diag(d)
    cov = B @ F @ B.T + D
    cov = 0.5 * (cov + cov.T)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "name": "factor_covariance",
        "method": "factor_bfb_d",
        "matrix": cov.tolist(),
        "shape": list(cov.shape),
        "n_obs": int(n_obs),
        "n_assets": int(n),
        "n_factors": int(k),
        "factor_cov": F.tolist(),
        "residual_vars": d.tolist(),
        "factor_method": factor_method,
        "residual_method": residual_method,
        "version": version,
    }
