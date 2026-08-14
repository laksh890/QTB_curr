"""Shared OLS / CSS fitting primitives for classical time-series models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize


@dataclass(slots=True)
class FitResult:
    params: np.ndarray
    residuals: np.ndarray
    fitted: np.ndarray
    sigma2: float
    loglik: float
    nobs: int
    k_params: int
    intercept: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.k_params

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.k_params * np.log(max(self.nobs, 1))

    @property
    def hqic(self) -> float:
        return -2 * self.loglik + 2 * self.k_params * np.log(np.log(max(self.nobs, 3)))

    @property
    def aicc(self) -> float:
        k, n = self.k_params, self.nobs
        if n - k - 1 <= 0:
            return self.aic
        return self.aic + (2 * k * (k + 1)) / (n - k - 1)


def lag_design(y: np.ndarray, p: int, *, intercept: bool = True) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    p = max(int(p), 0)
    if p == 0:
        Y = x.copy()
        X = np.ones((x.size, 1)) if intercept else np.zeros((x.size, 0))
        return X, Y
    n = x.size
    Y = x[p:]
    cols = [x[p - k : n - k] for k in range(1, p + 1)]
    X = np.column_stack(cols)
    if intercept:
        X = np.column_stack([np.ones(Y.size), X])
    return X, Y


def fit_ar_ols(y: np.ndarray, p: int, *, intercept: bool = True) -> FitResult:
    X, Y = lag_design(y, p, intercept=intercept)
    if Y.size == 0:
        return FitResult(np.zeros(p), np.array([]), np.array([]), 1.0, -1e9, 0, p + int(intercept))
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    fitted = X @ beta
    resid = Y - fitted
    nobs = int(Y.size)
    k = int(beta.size)
    sigma2 = float(np.dot(resid, resid) / max(nobs - k, 1))
    loglik = _gaussian_loglik(resid, sigma2)
    c = float(beta[0]) if intercept else 0.0
    phi = np.asarray(beta[1:] if intercept else beta, dtype=np.float64)
    return FitResult(
        params=phi,
        residuals=resid,
        fitted=fitted,
        sigma2=max(sigma2, 1e-12),
        loglik=loglik,
        nobs=nobs,
        k_params=k,
        intercept=c,
    )


def arma_innovations(
    y: np.ndarray,
    ar: np.ndarray,
    ma: np.ndarray,
    *,
    intercept: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional sum-of-squares residuals for ARMA."""
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    phi = np.asarray(ar, dtype=np.float64).reshape(-1)
    theta = np.asarray(ma, dtype=np.float64).reshape(-1)
    p, q = phi.size, theta.size
    n = x.size
    e = np.zeros(n, dtype=np.float64)
    fitted = np.zeros(n, dtype=np.float64)
    for t in range(n):
        ar_term = 0.0
        for i in range(min(p, t)):
            ar_term += phi[i] * x[t - 1 - i]
        ma_term = 0.0
        for j in range(min(q, t)):
            ma_term += theta[j] * e[t - 1 - j]
        fitted[t] = intercept + ar_term + ma_term
        # clip to keep CSS optimization numerically stable
        if not np.isfinite(fitted[t]):
            fitted[t] = x[t]
        e[t] = x[t] - fitted[t]
        if not np.isfinite(e[t]) or abs(e[t]) > 1e6:
            e[t] = float(np.clip(e[t] if np.isfinite(e[t]) else 0.0, -1e6, 1e6))
    return e, fitted


def fit_arma_css(
    y: np.ndarray,
    p: int,
    q: int,
    *,
    intercept: bool = True,
) -> FitResult:
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    p, q = max(int(p), 0), max(int(q), 0)
    if p == 0 and q == 0:
        mu = float(np.mean(x)) if intercept else 0.0
        resid = x - mu
        sigma2 = float(np.var(resid)) or 1e-12
        return FitResult(
            params=np.array([], dtype=np.float64),
            residuals=resid,
            fitted=np.full_like(x, mu),
            sigma2=sigma2,
            loglik=_gaussian_loglik(resid, sigma2),
            nobs=x.size,
            k_params=int(intercept),
            intercept=mu,
        )
    # init AR via OLS when possible
    ar0 = fit_ar_ols(x, max(p, 1), intercept=intercept).params if p else np.zeros(0)
    if ar0.size < p:
        ar0 = np.resize(ar0, p)
    theta0 = np.zeros(q)
    c0 = float(np.mean(x)) if intercept else 0.0
    x0 = np.concatenate([[c0] if intercept else [], ar0[:p], theta0])

    def objective(theta: np.ndarray) -> float:
        idx = 0
        c = float(theta[0]) if intercept else 0.0
        if intercept:
            idx = 1
        ar = np.clip(theta[idx : idx + p], -1.5, 1.5)
        ma = np.clip(theta[idx + p : idx + p + q], -1.5, 1.5)
        e, _ = arma_innovations(x, ar, ma, intercept=c)
        val = float(np.dot(e, e))
        return val if np.isfinite(val) else 1e20

    bounds = [(-10.0, 10.0)] * x0.size
    for i in range((1 if intercept else 0), x0.size):
        bounds[i] = (-1.5, 1.5)
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    theta = res.x if res.success else x0
    idx = 0
    c = float(theta[0]) if intercept else 0.0
    if intercept:
        idx = 1
    ar = np.asarray(theta[idx : idx + p], dtype=np.float64)
    ma = np.asarray(theta[idx + p : idx + p + q], dtype=np.float64)
    e, fitted = arma_innovations(x, ar, ma, intercept=c)
    nobs = int(x.size)
    k = p + q + int(intercept)
    sigma2 = float(np.dot(e, e) / max(nobs - k, 1))
    return FitResult(
        params=np.concatenate([ar, ma]),
        residuals=e,
        fitted=fitted,
        sigma2=max(sigma2, 1e-12),
        loglik=_gaussian_loglik(e, sigma2),
        nobs=nobs,
        k_params=k,
        intercept=c,
        metadata={"ar": ar.tolist(), "ma": ma.tolist(), "success": bool(res.success)},
    )


def forecast_arma(
    history: np.ndarray,
    residuals: np.ndarray,
    ar: np.ndarray,
    ma: np.ndarray,
    *,
    intercept: float = 0.0,
    horizon: int = 1,
) -> np.ndarray:
    hist = list(np.asarray(history, dtype=np.float64).reshape(-1))
    errs = list(np.asarray(residuals, dtype=np.float64).reshape(-1))
    phi = np.asarray(ar, dtype=np.float64).reshape(-1)
    theta = np.asarray(ma, dtype=np.float64).reshape(-1)
    out = np.empty(max(int(horizon), 1), dtype=np.float64)
    for h in range(out.size):
        ar_term = sum(phi[i] * hist[-1 - i] for i in range(min(phi.size, len(hist))))
        ma_term = sum(theta[j] * errs[-1 - j] for j in range(min(theta.size, len(errs))))
        yhat = intercept + ar_term + ma_term
        if not np.isfinite(yhat):
            yhat = hist[-1] if hist else 0.0
        out[h] = float(yhat)
        hist.append(out[h])
        errs.append(0.0)  # future innovations zero
    return out


def fit_var_ols(Y: np.ndarray, p: int, *, intercept: bool = True) -> dict[str, Any]:
    """Fit VAR(p) by equation-wise OLS. ``Y`` shape (T, K)."""
    data = np.asarray(Y, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    T, K = data.shape
    p = max(int(p), 1)
    if p + 2 >= T:
        return {
            "coefs": np.zeros((p, K, K)),
            "intercept": np.zeros(K),
            "residuals": np.zeros((0, K)),
            "fitted": np.zeros((0, K)),
            "sigma": np.eye(K),
            "nobs": 0,
            "k_params": 0,
            "loglik": -1e9,
        }
    # rows t=p..T-1
    target = data[p:]
    cols = []
    if intercept:
        cols.append(np.ones((T - p, 1)))
    for lag in range(1, p + 1):
        cols.append(data[p - lag : T - lag])
    X = np.concatenate(cols, axis=1)
    B, *_ = np.linalg.lstsq(X, target, rcond=None)  # (1+Kp, K)
    fitted = X @ B
    resid = target - fitted
    nobs = int(target.shape[0])
    sigma = (resid.T @ resid) / max(nobs - X.shape[1], 1)
    sigma = np.atleast_2d(sigma)
    # unpack coefficients
    offset = 1 if intercept else 0
    coefs = np.zeros((p, K, K), dtype=np.float64)
    for lag in range(p):
        block = B[offset + lag * K : offset + (lag + 1) * K, :]
        coefs[lag] = block.T  # K x K: y_t depends on y_{t-lag}
    intercept_v = B[0, :] if intercept else np.zeros(K)
    # Gaussian loglik
    sign, logdet = np.linalg.slogdet(sigma + 1e-12 * np.eye(K))
    if sign <= 0:
        logdet = 0.0
    ll = -0.5 * nobs * (K * np.log(2 * np.pi) + logdet) - 0.5 * float(
        np.sum(resid @ np.linalg.pinv(sigma) * resid)
    )
    k_params = int(B.size)
    return {
        "coefs": coefs,
        "intercept": np.asarray(intercept_v, dtype=np.float64),
        "residuals": resid,
        "fitted": fitted,
        "sigma": sigma,
        "nobs": nobs,
        "k_params": k_params,
        "loglik": float(ll),
        "B": B,
    }


def forecast_var(
    history: np.ndarray,
    coefs: np.ndarray,
    intercept: np.ndarray,
    *,
    horizon: int = 1,
) -> np.ndarray:
    hist = list(np.asarray(history, dtype=np.float64))
    if hist and not isinstance(hist[0], (list, np.ndarray)):
        # univariate list of scalars → wrap
        hist = [np.asarray(history[-1:], dtype=np.float64)]
    else:
        hist = [np.asarray(h, dtype=np.float64).reshape(-1) for h in np.asarray(history)]
    p, K, _ = coefs.shape
    out = np.empty((max(int(horizon), 1), K), dtype=np.float64)
    for h in range(out.shape[0]):
        yhat = intercept.copy()
        for lag in range(p):
            if lag < len(hist):
                yhat = yhat + coefs[lag] @ hist[-1 - lag]
        out[h] = yhat
        hist.append(yhat)
    return out


def _gaussian_loglik(resid: np.ndarray, sigma2: float) -> float:
    e = np.asarray(resid, dtype=np.float64).reshape(-1)
    n = e.size
    s2 = max(float(sigma2), 1e-300)
    return float(-0.5 * n * (np.log(2 * np.pi) + np.log(s2)) - 0.5 * np.dot(e, e) / s2)


def information_criteria(loglik: float, k: int, n: int) -> dict[str, float]:
    aic = -2 * loglik + 2 * k
    bic = -2 * loglik + k * np.log(max(n, 1))
    hqic = -2 * loglik + 2 * k * np.log(np.log(max(n, 3)))
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else aic
    return {"aic": float(aic), "aicc": float(aicc), "bic": float(bic), "hqic": float(hqic)}
