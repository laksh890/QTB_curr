"""Residual diagnostics for statistical forecasting models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass(slots=True)
class DiagnosticReport:
    nobs: int
    mean: float
    std: float
    skewness: float
    kurtosis: float
    durbin_watson: float
    ljung_box_stat: float
    ljung_box_pvalue: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    arch_lm_stat: float
    arch_lm_pvalue: float
    acf: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nobs": self.nobs,
            "mean": self.mean,
            "std": self.std,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "durbin_watson": self.durbin_watson,
            "ljung_box_stat": self.ljung_box_stat,
            "ljung_box_pvalue": self.ljung_box_pvalue,
            "jarque_bera_stat": self.jarque_bera_stat,
            "jarque_bera_pvalue": self.jarque_bera_pvalue,
            "arch_lm_stat": self.arch_lm_stat,
            "arch_lm_pvalue": self.arch_lm_pvalue,
            "acf": list(self.acf),
            "metadata": dict(self.metadata),
        }


def durbin_watson(resid: np.ndarray) -> float:
    e = np.asarray(resid, dtype=np.float64).reshape(-1)
    if e.size < 2:
        return float("nan")
    return float(np.sum(np.diff(e) ** 2) / max(np.sum(e**2), 1e-300))


def acf(resid: np.ndarray, *, nlags: int = 20) -> list[float]:
    e = np.asarray(resid, dtype=np.float64).reshape(-1)
    e = e - np.mean(e)
    n = e.size
    if n < 2:
        return []
    var = float(np.dot(e, e))
    if var <= 1e-300:
        return [0.0] * min(nlags, n - 1)
    out = []
    for lag in range(1, min(nlags, n - 1) + 1):
        out.append(float(np.dot(e[:-lag], e[lag:]) / var))
    return out


def ljung_box(resid: np.ndarray, *, lags: int = 10) -> tuple[float, float]:
    e = np.asarray(resid, dtype=np.float64).reshape(-1)
    n = e.size
    rhos = acf(e, nlags=lags)
    if not rhos:
        return 0.0, 1.0
    qb = n * (n + 2) * sum((r**2) / (n - i - 1) for i, r in enumerate(rhos))
    p = float(1.0 - stats.chi2.cdf(qb, df=len(rhos)))
    return float(qb), p


def jarque_bera(resid: np.ndarray) -> tuple[float, float]:
    e = np.asarray(resid, dtype=np.float64).reshape(-1)
    n = e.size
    if n < 3:
        return 0.0, 1.0
    e = e - np.mean(e)
    s = float(np.mean(e**3) / max(np.std(e) ** 3, 1e-300))
    k = float(np.mean(e**4) / max(np.std(e) ** 4, 1e-300) - 3.0)
    jb = n / 6.0 * (s**2 + 0.25 * k**2)
    p = float(1.0 - stats.chi2.cdf(jb, df=2))
    return float(jb), p


def arch_lm(resid: np.ndarray, *, lags: int = 5) -> tuple[float, float]:
    """Engle ARCH-LM heteroskedasticity test on squared residuals."""
    e2 = np.asarray(resid, dtype=np.float64).reshape(-1) ** 2
    n = e2.size
    L = max(int(lags), 1)
    if n <= L + 2:
        return 0.0, 1.0
    y = e2[L:]
    X = np.column_stack([np.ones(n - L)] + [e2[L - i : n - i] for i in range(1, L + 1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - fitted) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    lm = (n - L) * r2
    p = float(1.0 - stats.chi2.cdf(lm, df=L))
    return float(lm), p


def run_diagnostics(resid: np.ndarray, *, nlags: int = 15) -> DiagnosticReport:
    e = np.asarray(resid, dtype=np.float64).reshape(-1)
    n = e.size
    if n == 0:
        return DiagnosticReport(
            nobs=0,
            mean=0.0,
            std=0.0,
            skewness=0.0,
            kurtosis=0.0,
            durbin_watson=float("nan"),
            ljung_box_stat=0.0,
            ljung_box_pvalue=1.0,
            jarque_bera_stat=0.0,
            jarque_bera_pvalue=1.0,
            arch_lm_stat=0.0,
            arch_lm_pvalue=1.0,
            acf=[],
        )
    z = e - np.mean(e)
    sd = float(np.std(e)) or 1e-12
    skew = float(np.mean((z / sd) ** 3))
    kurt = float(np.mean((z / sd) ** 4) - 3.0)
    lb_s, lb_p = ljung_box(e, lags=min(nlags, max(n // 5, 1)))
    jb_s, jb_p = jarque_bera(e)
    arch_s, arch_p = arch_lm(e, lags=min(5, max(n // 10, 1)))
    return DiagnosticReport(
        nobs=n,
        mean=float(np.mean(e)),
        std=sd,
        skewness=skew,
        kurtosis=kurt,
        durbin_watson=durbin_watson(e),
        ljung_box_stat=lb_s,
        ljung_box_pvalue=lb_p,
        jarque_bera_stat=jb_s,
        jarque_bera_pvalue=jb_p,
        arch_lm_stat=arch_s,
        arch_lm_pvalue=arch_p,
        acf=acf(e, nlags=nlags),
    )
