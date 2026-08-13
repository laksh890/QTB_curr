"""OLS residualization of signals against factor exposures."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _as_panel(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected 1D or 2D array, got shape {arr.shape}")
    return arr


def _ols_residuals(y: np.ndarray, X: np.ndarray, *, add_intercept: bool = True) -> np.ndarray:
    """Residualize ``y`` (n,) on ``X`` (n, k). Returns y - Xb."""
    mask = np.isfinite(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mask &= np.all(np.isfinite(X), axis=1)
    resid = np.full_like(y, np.nan, dtype=np.float64)
    if int(mask.sum()) < X.shape[1] + (1 if add_intercept else 0) + 1:
        return resid
    Xm = X[mask]
    ym = y[mask]
    if add_intercept:
        Xm = np.column_stack([np.ones(Xm.shape[0]), Xm])
    try:
        coef, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    except np.linalg.LinAlgError:
        return resid
    pred = Xm @ coef
    resid[mask] = ym - pred
    return resid


def residualize_vs_factors(
    signal: Any,
    factors: Any,
    *,
    add_intercept: bool = True,
) -> np.ndarray:
    """Residualize each date's cross-section against factor loadings.

    Parameters
    ----------
    signal :
        ``(T, N)`` signal panel.
    factors :
        ``(T, N, K)`` factor exposures, or ``(N, K)`` static exposures broadcast
        across time, or ``(T, N)`` single-factor panel.
    """
    panel = _as_panel(signal)
    t, n = panel.shape
    f = np.asarray(factors, dtype=np.float64)

    if f.ndim == 2:
        if f.shape == (t, n):
            f = f.reshape(t, n, 1)
        elif f.shape[0] == n:
            f = np.broadcast_to(f.reshape(1, n, f.shape[1]), (t, n, f.shape[1])).copy()
        else:
            raise ValueError(f"factors shape {f.shape} incompatible with signal {panel.shape}")
    elif f.ndim == 3:
        if f.shape[0] != t or f.shape[1] != n:
            raise ValueError(f"factors shape {f.shape} incompatible with signal {panel.shape}")
    else:
        raise ValueError(f"factors must be 2D or 3D, got {f.ndim}D")

    out = np.full_like(panel, np.nan)
    for i in range(t):
        out[i, :] = _ols_residuals(panel[i, :], f[i], add_intercept=add_intercept)
    return out


def residualize_vs_signals(
    signal: Any,
    controls: Sequence[Any] | dict[str, Any],
    *,
    add_intercept: bool = True,
) -> np.ndarray:
    """Residualize ``signal`` against other alpha / control signal panels."""
    panel = _as_panel(signal)
    if isinstance(controls, dict):
        mats = [_as_panel(v) for v in controls.values()]
    else:
        mats = [_as_panel(v) for v in controls]
    if not mats:
        return panel.copy()
    for m in mats:
        if m.shape != panel.shape:
            raise ValueError(f"control shape {m.shape} != signal shape {panel.shape}")
    stacked = np.stack(mats, axis=-1)  # (T, N, K)
    return residualize_vs_factors(panel, stacked, add_intercept=add_intercept)


def beta_residualize(
    signal: Any,
    market_returns: Any,
    asset_returns: Any,
    *,
    lookback: int = 60,
    min_obs: int = 20,
) -> np.ndarray:
    """Residualize signal using rolling OLS betas of assets vs market.

    ``market_returns`` is ``(T,)``, ``asset_returns`` is ``(T, N)``.
    Beta at t uses returns in ``[t-lookback, t)`` (no look-ahead).
    """
    panel = _as_panel(signal)
    mkt = np.asarray(market_returns, dtype=np.float64).reshape(-1)
    rets = _as_panel(asset_returns)
    t, n = panel.shape
    if mkt.shape[0] != t or rets.shape != (t, n):
        raise ValueError("market_returns / asset_returns shape mismatch with signal")

    betas = np.full((t, n), np.nan, dtype=np.float64)
    lb = max(2, int(lookback))
    for i in range(t):
        start = max(0, i - lb)
        if i - start < min_obs:
            continue
        y_win = rets[start:i]
        x_win = mkt[start:i]
        for j in range(n):
            y = y_win[:, j]
            mask = np.isfinite(y) & np.isfinite(x_win)
            if int(mask.sum()) < min_obs:
                continue
            xf = x_win[mask]
            yf = y[mask]
            var = float(np.var(xf))
            if var < 1e-18:
                continue
            betas[i, j] = float(np.cov(yf, xf, ddof=1)[0, 1] / var)

    return residualize_vs_factors(panel, betas, add_intercept=True)
