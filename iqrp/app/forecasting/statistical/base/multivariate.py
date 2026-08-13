"""Cointegration, Granger causality, IRF, and FEVD utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from iqrp.app.forecasting.statistical.base.fitting import fit_var_ols, lag_design
from iqrp.app.forecasting.statistical.base.stationarity import adf_test


@dataclass(slots=True)
class CointegrationResult:
    method: str
    statistic: float
    pvalue: float
    rank: int
    eigenvectors: np.ndarray | None = None
    eigenvalues: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "rank": self.rank,
            "eigenvectors": None if self.eigenvectors is None else self.eigenvectors.tolist(),
            "eigenvalues": None if self.eigenvalues is None else self.eigenvalues.tolist(),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class GrangerResult:
    cause: int
    effect: int
    f_stat: float
    pvalue: float
    lag: int
    significant: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "f_stat": self.f_stat,
            "pvalue": self.pvalue,
            "lag": self.lag,
            "significant": self.significant,
            "metadata": dict(self.metadata),
        }


def engle_granger(y: np.ndarray, x: np.ndarray) -> CointegrationResult:
    """Engle–Granger two-step cointegration test (bivariate)."""
    a = np.asarray(y, dtype=np.float64).reshape(-1)
    b = np.asarray(x, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    X = np.column_stack([np.ones(n), b])
    beta, *_ = np.linalg.lstsq(X, a, rcond=None)
    resid = a - X @ beta
    adf = adf_test(resid, regression="n")
    # Engle-Granger critical approx for constant
    crit5 = -3.37
    return CointegrationResult(
        method="engle_granger",
        statistic=adf.statistic,
        pvalue=adf.pvalue,
        rank=1 if adf.statistic < crit5 else 0,
        metadata={"beta": beta.tolist(), "critical_5%": crit5},
    )


def johansen_trace(Y: np.ndarray, *, lags: int = 1) -> CointegrationResult:
    """Simplified Johansen trace test via reduced-rank VECM eigenvalues."""
    data = np.asarray(Y, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    T, K = data.shape
    p = max(int(lags), 1)
    if T <= p + K + 2 or K < 2:
        return CointegrationResult("johansen", 0.0, 1.0, 0)
    # ΔY_t and Y_{t-1}
    dY = np.diff(data, axis=0)
    Ylag = data[:-1]
    # residualize vs lagged differences
    if p > 1:
        # project out ΔY lags
        rows = dY.shape[0] - (p - 1)
        dY_t = dY[p - 1 :]
        Y_l = Ylag[p - 1 :]
        Z = np.column_stack([dY[p - 1 - i : p - 1 - i + rows] for i in range(1, p)])
        # residualize
        Bz, *_ = np.linalg.lstsq(Z, dY_t, rcond=None)
        R0 = dY_t - Z @ Bz
        By, *_ = np.linalg.lstsq(Z, Y_l, rcond=None)
        R1 = Y_l - Z @ By
    else:
        R0, R1 = dY, Ylag
    S00 = R0.T @ R0 / R0.shape[0]
    S11 = R1.T @ R1 / R1.shape[0]
    S01 = R0.T @ R1 / R0.shape[0]
    S10 = S01.T
    try:
        S11_inv = np.linalg.pinv(S11)
        M = S11_inv @ S10 @ np.linalg.pinv(S00) @ S01
        eigvals = np.sort(np.real(np.linalg.eigvals(M)))[::-1]
        eigvals = np.clip(eigvals, 0.0, 0.999999)
    except Exception:  # noqa: BLE001
        return CointegrationResult("johansen", 0.0, 1.0, 0)
    # trace for r=0
    nobs = R0.shape[0]
    trace = float(-nobs * np.sum(np.log(1.0 - eigvals)))
    # critical approx for K=2 r=0 ~ 15.49 at 5%
    crit = 15.49 if K == 2 else 29.8
    rank = int(np.sum(eigvals > 0.05))
    pvalue = float(np.exp(-trace / crit))
    return CointegrationResult(
        method="johansen",
        statistic=trace,
        pvalue=min(max(pvalue, 0.0), 1.0),
        rank=min(rank, K - 1),
        eigenvalues=eigvals,
        metadata={"critical_5%": crit, "nobs": nobs},
    )


def granger_causality(
    Y: np.ndarray,
    *,
    cause: int,
    effect: int,
    lag: int = 2,
    alpha: float = 0.05,
) -> GrangerResult:
    """Granger causality test: does ``cause`` help predict ``effect``?"""
    data = np.asarray(Y, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    T, K = data.shape
    p = max(int(lag), 1)
    if cause >= K or effect >= K or T <= p + 5:
        return GrangerResult(cause, effect, 0.0, 1.0, p, False)
    y = data[p:, effect]
    # restricted: lags of effect only
    X_r = np.column_stack([data[p - k : T - k, effect] for k in range(1, p + 1)])
    X_r = np.column_stack([np.ones(y.size), X_r])
    # unrestricted: + lags of cause
    X_u = np.column_stack(
        [X_r] + [data[p - k : T - k, cause] for k in range(1, p + 1)]
    )
    Br, *_ = np.linalg.lstsq(X_r, y, rcond=None)
    Bu, *_ = np.linalg.lstsq(X_u, y, rcond=None)
    ssr_r = float(np.sum((y - X_r @ Br) ** 2))
    ssr_u = float(np.sum((y - X_u @ Bu) ** 2))
    q = p
    df_u = max(y.size - X_u.shape[1], 1)
    f_stat = ((ssr_r - ssr_u) / q) / max(ssr_u / df_u, 1e-300)
    pvalue = float(1.0 - stats.f.cdf(f_stat, q, df_u))
    return GrangerResult(
        cause=cause,
        effect=effect,
        f_stat=float(f_stat),
        pvalue=pvalue,
        lag=p,
        significant=pvalue < alpha,
    )


def impulse_response(
    coefs: np.ndarray,
    sigma: np.ndarray,
    *,
    horizon: int = 10,
    orthogonal: bool = True,
) -> np.ndarray:
    """
    Impulse response functions.

    Parameters
    ----------
    coefs : (p, K, K)
    sigma : (K, K)

    Returns
    -------
    irf : (horizon, K, K)  response of row variable to column shock
    """
    p, K, _ = coefs.shape
    H = max(int(horizon), 1)
    # companion MA coefficients Psi
    Psi = np.zeros((H, K, K), dtype=np.float64)
    Psi[0] = np.eye(K)
    for h in range(1, H):
        acc = np.zeros((K, K))
        for lag in range(min(p, h)):
            acc = acc + coefs[lag] @ Psi[h - 1 - lag]
        Psi[h] = acc
    if orthogonal:
        try:
            P = np.linalg.cholesky(sigma + 1e-12 * np.eye(K))
        except np.linalg.LinAlgError:
            P = np.eye(K)
        for h in range(H):
            Psi[h] = Psi[h] @ P
    return Psi


def fevd(
    coefs: np.ndarray,
    sigma: np.ndarray,
    *,
    horizon: int = 10,
) -> np.ndarray:
    """Forecast error variance decomposition, shape ``(horizon, K, K)``."""
    irf = impulse_response(coefs, sigma, horizon=horizon, orthogonal=True)
    H, K, _ = irf.shape
    out = np.zeros((H, K, K), dtype=np.float64)
    # cumulative contribution of each shock to MSE of each variable
    mse = np.zeros(K)
    contrib = np.zeros((K, K))
    for h in range(H):
        for j in range(K):  # shock
            for i in range(K):  # variable
                contrib[i, j] += irf[h, i, j] ** 2
        mse = contrib.sum(axis=1)
        for i in range(K):
            denom = max(mse[i], 1e-300)
            out[h, i, :] = contrib[i, :] / denom
    return out


def fit_vecm_engle_granger(
    Y: np.ndarray, *, lags: int = 1
) -> dict[str, Any]:
    """Bivariate / multi Engle–Granger VECM: Δy = α β' y_{t-1} + lags + e."""
    data = np.asarray(Y, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    T, K = data.shape
    p = max(int(lags), 1)
    dY = np.diff(data, axis=0)
    Ylag = data[:-1]
    # first eigenvector from johansen as β
    joh = johansen_trace(data, lags=p)
    if joh.eigenvalues is not None and joh.eigenvalues.size:
        # reconstruct beta approx via OLS of first series on others
        if K >= 2:
            X = np.column_stack([np.ones(T), data[:, 1:]])
            b, *_ = np.linalg.lstsq(X, data[:, 0], rcond=None)
            beta = np.ones(K)
            beta[0] = 1.0
            beta[1:] = -b[1:]
        else:
            beta = np.array([1.0])
    else:
        beta = np.ones(K)
        beta[0] = 1.0
    ect = Ylag @ beta  # (T-1,)
    rows = dY.shape[0] - (p - 1) if p > 1 else dY.shape[0]
    start = p - 1 if p > 1 else 0
    dep = dY[start:]
    ect_t = ect[start:]
    cols = [ect_t.reshape(-1, 1)]
    for lag in range(1, p):
        cols.append(dY[start - lag : start - lag + rows])
    cols.append(np.ones((rows, 1)))
    X = np.concatenate(cols, axis=1)
    B, *_ = np.linalg.lstsq(X, dep, rcond=None)
    fitted = X @ B
    resid = dep - fitted
    alpha = B[0, :]  # loading
    return {
        "beta": beta,
        "alpha": alpha,
        "B": B,
        "residuals": resid,
        "fitted": fitted,
        "nobs": int(rows),
        "rank": joh.rank,
        "johansen": joh,
    }
