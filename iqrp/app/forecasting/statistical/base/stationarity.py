"""Stationarity tests and differencing / transform utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats


@dataclass(slots=True)
class StationarityResult:
    statistic: float
    pvalue: float
    used_lags: int
    nobs: int
    critical_values: dict[str, float]
    stationary: bool
    method: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "used_lags": self.used_lags,
            "nobs": self.nobs,
            "critical_values": dict(self.critical_values),
            "stationary": self.stationary,
            "method": self.method,
            "metadata": dict(self.metadata or {}),
        }


def difference(y: np.ndarray, *, order: int = 1) -> np.ndarray:
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    out = x
    for _ in range(max(int(order), 0)):
        out = np.diff(out)
    return out


def seasonal_difference(y: np.ndarray, *, period: int, order: int = 1) -> np.ndarray:
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    out = x
    s = max(int(period), 1)
    for _ in range(max(int(order), 0)):
        if out.size <= s:
            return np.array([], dtype=np.float64)
        out = out[s:] - out[:-s]
    return out


def integrate(diffed: np.ndarray, history: np.ndarray, *, order: int = 1) -> np.ndarray:
    """Undo differencing using trailing ``history`` levels."""
    x = np.asarray(diffed, dtype=np.float64).reshape(-1)
    hist = np.asarray(history, dtype=np.float64).reshape(-1)
    if order <= 0:
        return x.copy()
    if order == 1:
        level = float(hist[-1]) if hist.size else 0.0
        out = np.empty(x.size, dtype=np.float64)
        for i, d in enumerate(x):
            level = level + float(d)
            out[i] = level
        return out
    # higher order: recursive
    prev = integrate(x, difference(hist, order=order - 1), order=1)
    return integrate(prev, hist, order=order - 1)


def seasonal_integrate(
    diffed: np.ndarray, history: np.ndarray, *, period: int, order: int = 1
) -> np.ndarray:
    x = np.asarray(diffed, dtype=np.float64).reshape(-1)
    hist = np.asarray(history, dtype=np.float64).reshape(-1)
    s = max(int(period), 1)
    out = x.copy()
    for _ in range(max(int(order), 0)):
        rebuilt = np.empty(out.size, dtype=np.float64)
        buf = list(hist[-s:]) if hist.size >= s else list(hist) + [0.0] * (s - hist.size)
        for i, d in enumerate(out):
            val = float(buf[-s]) + float(d)
            rebuilt[i] = val
            buf.append(val)
        hist = np.asarray(buf, dtype=np.float64)
        out = rebuilt
    return out


def log_transform(y: np.ndarray, *, offset: float = 0.0) -> np.ndarray:
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    return np.log(np.clip(x + offset, 1e-300, None))


def box_cox(y: np.ndarray, *, lam: float = 0.0, offset: float = 0.0) -> np.ndarray:
    x = np.asarray(y, dtype=np.float64).reshape(-1) + offset
    x = np.clip(x, 1e-300, None)
    if abs(lam) < 1e-12:
        return np.log(x)
    return (np.power(x, lam) - 1.0) / lam


def _lag_matrix(y: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    n = x.size
    L = max(int(lags), 0)
    if L == 0 or n <= L + 1:  # pragma: no cover
        return np.zeros((0, max(L, 1))), np.zeros(0)
    rows = n - L
    Y = x[L:]
    X = np.column_stack([x[L - k : n - k] for k in range(1, L + 1)])
    return X[:rows], Y[:rows]


def adf_test(
    y: np.ndarray,
    *,
    max_lags: int | None = None,
    regression: Literal["c", "ct", "n"] = "c",
    alpha: float = 0.05,
) -> StationarityResult:
    """Augmented Dickey–Fuller unit-root test (approximation to MacKinnon p-values)."""
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 10:
        return StationarityResult(0.0, 1.0, 0, n, {}, False, "adf")
    dy = np.diff(x)
    max_l = int(max_lags if max_lags is not None else max(1, int(np.floor(12 * (n / 100) ** 0.25))))
    max_l = min(max_l, n // 3)
    best_aic = np.inf
    best = None
    for lag in range(0, max_l + 1):
        # Δy_t = α + β y_{t-1} + Σ γ_i Δy_{t-i}
        y_level = x[1:]
        end = dy.size
        start = lag
        if end - start < lag + 5:
            continue
        dep = dy[start:end]
        level = y_level[start:end]
        cols = [level]
        if regression in {"c", "ct"}:
            cols.append(np.ones(dep.size))
        if regression == "ct":
            cols.append(np.arange(start + 1, end + 1, dtype=np.float64))
        for i in range(1, lag + 1):
            cols.append(dy[start - i : end - i])
        X = np.column_stack(cols)
        try:
            beta, *_ = np.linalg.lstsq(X, dep, rcond=None)
        except Exception:
            continue
        resid = dep - X @ beta
        s2 = float(np.dot(resid, resid) / max(dep.size - X.shape[1], 1))
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(max(s2 * xtx_inv[0, 0], 1e-300))
        tstat = float(beta[0] / se)
        aic = dep.size * np.log(max(s2, 1e-300)) + 2 * X.shape[1]
        if aic < best_aic:
            best_aic = aic
            best = (tstat, lag, dep.size, s2)
    if best is None:
        return StationarityResult(0.0, 1.0, 0, n, {}, False, "adf")
    tstat, used_lags, nobs, _ = best
    # MacKinnon approx critical values for constant case
    crit = (
        {"1%": -3.43, "5%": -2.86, "10%": -2.57}
        if regression != "n"
        else {"1%": -2.58, "5%": -1.95, "10%": -1.62}
    )
    if regression == "ct":
        crit = {"1%": -3.96, "5%": -3.41, "10%": -3.13}
    # rough p-value from normal for ranking (conservative)
    pvalue = float(stats.norm.cdf(tstat))
    return StationarityResult(
        statistic=tstat,
        pvalue=pvalue,
        used_lags=used_lags,
        nobs=nobs,
        critical_values=crit,
        stationary=tstat < crit["5%"],
        method="adf",
        metadata={"alpha": alpha, "regression": regression},
    )


def kpss_test(
    y: np.ndarray,
    *,
    regression: Literal["c", "ct"] = "c",
    lags: int | None = None,
    alpha: float = 0.05,
) -> StationarityResult:
    """KPSS level/trend stationarity test."""
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 10:
        return StationarityResult(0.0, 1.0, 0, n, {}, True, "kpss")
    t = np.arange(1, n + 1, dtype=np.float64)
    if regression == "ct":
        X = np.column_stack([np.ones(n), t])
    else:
        X = np.ones((n, 1))
    beta, *_ = np.linalg.lstsq(X, x, rcond=None)
    resid = x - X @ beta
    s = np.cumsum(resid)
    l = int(lags if lags is not None else max(1, int(np.floor(4 * (n / 100) ** 0.25))))
    # Newey-West long-run variance
    gamma0 = float(np.dot(resid, resid) / n)
    lrv = gamma0
    for h in range(1, l + 1):
        w = 1.0 - h / (l + 1)
        gamma = float(np.dot(resid[h:], resid[:-h]) / n)
        lrv += 2 * w * gamma
    lrv = max(lrv, 1e-300)
    stat = float(np.sum(s**2) / (n**2 * lrv))
    crit = (
        {"10%": 0.347, "5%": 0.463, "1%": 0.739}
        if regression == "c"
        else {"10%": 0.119, "5%": 0.146, "1%": 0.216}
    )
    # approximate p: higher stat → reject stationarity
    pvalue = float(np.exp(-stat / crit["5%"]))
    return StationarityResult(
        statistic=stat,
        pvalue=min(max(pvalue, 0.0), 1.0),
        used_lags=l,
        nobs=n,
        critical_values=crit,
        stationary=stat < crit["5%"],
        method="kpss",
        metadata={"alpha": alpha, "regression": regression},
    )


def phillips_perron_test(
    y: np.ndarray,
    *,
    regression: Literal["c", "ct"] = "c",
    lags: int | None = None,
    alpha: float = 0.05,
) -> StationarityResult:
    """Phillips–Perron unit-root test with Newey–West correction."""
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 10:
        return StationarityResult(0.0, 1.0, 0, n, {}, False, "pp")
    y1 = x[1:]
    y0 = x[:-1]
    if regression == "ct":
        t = np.arange(1, y1.size + 1, dtype=np.float64)
        X = np.column_stack([y0, np.ones(y1.size), t])
    else:
        X = np.column_stack([y0, np.ones(y1.size)])
    beta, *_ = np.linalg.lstsq(X, y1, rcond=None)
    resid = y1 - X @ beta
    s2 = float(np.dot(resid, resid) / max(y1.size - X.shape[1], 1))
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(max(s2 * xtx_inv[0, 0], 1e-300))
    t_ols = float((beta[0] - 1.0) / se) if False else float(beta[0] / se)
    # use coefficient on lag level relative to unit root: rewrite as Δy
    # recompute as ADF style on levels without lags
    dy = y1 - y0
    Xd = X.copy()
    beta_d, *_ = np.linalg.lstsq(Xd, dy, rcond=None)
    resid = dy - Xd @ beta_d
    nobs = dy.size
    l = int(lags if lags is not None else max(1, int(np.floor(4 * (n / 100) ** 0.25))))
    gamma0 = float(np.dot(resid, resid) / nobs)
    lrv = gamma0
    for h in range(1, l + 1):
        w = 1.0 - h / (l + 1)
        lrv += 2 * w * float(np.dot(resid[h:], resid[:-h]) / nobs)
    lrv = max(lrv, 1e-300)
    # PP Z_t approximation
    se_b = np.sqrt(max(gamma0 * xtx_inv[0, 0], 1e-300))
    tstat = float(beta_d[0] / se_b)
    corr = 0.5 * (lrv - gamma0) * float(np.sqrt(xtx_inv[0, 0]) * nobs) / max(lrv, 1e-300)
    z_t = tstat * np.sqrt(gamma0 / lrv) - corr
    crit = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
    if regression == "ct":
        crit = {"1%": -3.96, "5%": -3.41, "10%": -3.13}
    pvalue = float(stats.norm.cdf(z_t))
    return StationarityResult(
        statistic=float(z_t),
        pvalue=pvalue,
        used_lags=l,
        nobs=nobs,
        critical_values=crit,
        stationary=z_t < crit["5%"],
        method="pp",
        metadata={"alpha": alpha, "ols_t": t_ols},
    )


def suggest_differencing(
    y: np.ndarray,
    *,
    max_d: int = 2,
    alpha: float = 0.05,
) -> int:
    """Choose d via successive ADF tests."""
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    for d in range(0, max_d + 1):
        series = difference(x, order=d) if d else x
        if series.size < 15:
            return max(d - 1, 0)
        res = adf_test(series, alpha=alpha)
        if res.stationary:
            return d
    return max_d


def suggest_seasonal_differencing(
    y: np.ndarray,
    *,
    period: int,
    max_D: int = 1,
    alpha: float = 0.05,
) -> int:
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    s = max(int(period), 1)
    for D in range(0, max_D + 1):
        series = seasonal_difference(x, period=s, order=D) if D else x
        if series.size < s + 15:
            return max(D - 1, 0)
        # seasonal strength heuristic + ADF
        if D == 0 and series.size > 2 * s:
            # ACF at seasonal lag
            z = series - np.mean(series)
            acf_s = float(np.dot(z[s:], z[:-s]) / max(np.dot(z, z), 1e-300))
            if abs(acf_s) < 0.3 and adf_test(series, alpha=alpha).stationary:
                return 0
        elif adf_test(series, alpha=alpha).stationary:
            return D
    return max_D
