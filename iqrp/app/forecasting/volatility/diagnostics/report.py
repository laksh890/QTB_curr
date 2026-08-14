"""Volatility diagnostic report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass(slots=True)
class VolatilityDiagnosticReport:
    standardized_residuals: list[float]
    arch_lm_stat: float
    arch_lm_pvalue: float
    ljung_box_stat: float
    ljung_box_pvalue: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    persistence: float
    half_life: float
    mean_variance: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "standardized_residuals": list(self.standardized_residuals[:200]),
            "arch_lm_stat": self.arch_lm_stat,
            "arch_lm_pvalue": self.arch_lm_pvalue,
            "ljung_box_stat": self.ljung_box_stat,
            "ljung_box_pvalue": self.ljung_box_pvalue,
            "jarque_bera_stat": self.jarque_bera_stat,
            "jarque_bera_pvalue": self.jarque_bera_pvalue,
            "persistence": self.persistence,
            "half_life": self.half_life,
            "mean_variance": self.mean_variance,
            "metadata": dict(self.metadata),
        }


def _ljung_box(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    n = x.size
    if n < lags + 2:
        return 0.0, 1.0
    x = x - np.mean(x)
    acf = np.correlate(x, x, mode="full")[n - 1 :]
    acf = acf / max(acf[0], 1e-300)
    q = n * (n + 2) * np.sum([(acf[k] ** 2) / (n - k) for k in range(1, lags + 1)])
    p = float(1 - stats.chi2.cdf(q, lags))
    return float(q), p


def _arch_lm(resid2: np.ndarray, lags: int = 5) -> tuple[float, float]:
    n = resid2.size
    if n <= lags + 1:
        return 0.0, 1.0
    y = resid2[lags:]
    X = np.column_stack([resid2[lags - i - 1 : n - i - 1] for i in range(lags)])
    X = np.column_stack([np.ones(y.size), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ beta
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - fitted) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-300)
        lm = float(y.size * r2)
        p = float(1 - stats.chi2.cdf(lm, lags))
        return lm, p
    except Exception:
        return 0.0, 1.0


def persistence_and_half_life(params: dict[str, float]) -> tuple[float, float]:
    alpha = float(params.get("alpha", params.get("alpha_0", 0.0)))
    beta = float(params.get("beta", params.get("beta_0", 0.0)))
    gamma = float(params.get("gamma", params.get("gamma_0", 0.0)))
    persist = alpha + beta + 0.5 * gamma
    if params.get("lambda") is not None:
        persist = float(params["lambda"])
    persist = float(np.clip(persist, 0.0, 0.999999))
    if persist <= 0:
        hl = 0.0
    else:
        hl = float(np.log(0.5) / np.log(persist))
    return persist, hl


def run_vol_diagnostics(
    returns: np.ndarray,
    variance: np.ndarray,
    *,
    params: dict[str, float] | None = None,
) -> VolatilityDiagnosticReport:
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    v = np.clip(np.asarray(variance, dtype=np.float64).reshape(-1), 1e-12, None)
    n = min(r.size, v.size)
    r, v = r[:n], v[:n]
    z = r / np.sqrt(v)
    z2 = z**2
    arch_stat, arch_p = _arch_lm(z2)
    lb_stat, lb_p = _ljung_box(z2)
    jb_stat, jb_p = stats.jarque_bera(z) if z.size >= 8 else (0.0, 1.0)
    persist, hl = persistence_and_half_life(params or {})
    return VolatilityDiagnosticReport(
        standardized_residuals=z.tolist(),
        arch_lm_stat=float(arch_stat),
        arch_lm_pvalue=float(arch_p),
        ljung_box_stat=float(lb_stat),
        ljung_box_pvalue=float(lb_p),
        jarque_bera_stat=float(jb_stat),
        jarque_bera_pvalue=float(jb_p),
        persistence=persist,
        half_life=hl,
        mean_variance=float(np.mean(v)),
        metadata={"n": int(n), "params": dict(params or {})},
    )
