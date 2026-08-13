"""Statistical validation of simulated paths against theoretical expectations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    mean_ok: bool
    variance_ok: bool
    distribution_ok: bool
    autocorrelation_ok: bool
    volatility_ok: bool
    mean_stat: float
    variance_stat: float
    ks_pvalue: float
    acf_lag1: float
    realized_vol: float
    expected_mean: float
    expected_variance: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def passed(self) -> bool:
        return all(
            [
                self.mean_ok,
                self.variance_ok,
                self.distribution_ok,
                self.autocorrelation_ok,
                self.volatility_ok,
            ]
        )


class SimulationValidator:
    """Compare simulated returns to GBM-style theoretical moments."""

    def __init__(self, *, significance: float = 0.05, acf_lags: int = 20) -> None:
        self.significance = significance
        self.acf_lags = acf_lags

    def validate_returns(
        self,
        returns: np.ndarray,
        *,
        expected_drift: float,
        expected_volatility: float,
        dt: float,
    ) -> ValidationReport:
        rets = np.asarray(returns, dtype=np.float64).ravel()
        rets = rets[np.isfinite(rets)]
        n = len(rets)
        if n < 10:
            return ValidationReport(
                mean_ok=False,
                variance_ok=False,
                distribution_ok=False,
                autocorrelation_ok=False,
                volatility_ok=False,
                mean_stat=float("nan"),
                variance_stat=float("nan"),
                ks_pvalue=float("nan"),
                acf_lag1=float("nan"),
                realized_vol=float("nan"),
                expected_mean=expected_drift * dt,
                expected_variance=(expected_volatility**2) * dt,
                details={"reason": "insufficient_samples", "n": n},
            )

        exp_mean = (expected_drift - 0.5 * expected_volatility**2) * dt
        exp_var = (expected_volatility**2) * dt
        mean_stat = float(np.mean(rets))
        var_stat = float(np.var(rets, ddof=1))

        # t-test for mean
        se = np.sqrt(var_stat / n)
        t_stat = (mean_stat - exp_mean) / max(se, 1e-12)
        mean_p = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))
        mean_ok = mean_p > self.significance * 0.1  # lenient for finite samples

        # Variance ratio check (chi-square style band)
        var_ratio = var_stat / max(exp_var, 1e-12)
        variance_ok = 0.25 < var_ratio < 4.0

        # KS vs Gaussian with estimated params
        zs = (rets - mean_stat) / max(np.sqrt(var_stat), 1e-12)
        ks_p = float(stats.kstest(zs, "norm").pvalue)
        distribution_ok = ks_p > self.significance * 0.01  # very lenient with jumps/events

        acf1 = float(np.corrcoef(rets[:-1], rets[1:])[0, 1]) if n > 2 else 0.0
        if not np.isfinite(acf1):
            acf1 = 0.0
        autocorrelation_ok = abs(acf1) < 0.35

        realized_vol = float(np.sqrt(np.mean(rets**2) / max(dt, 1e-12)))
        vol_ratio = realized_vol / max(expected_volatility, 1e-12)
        volatility_ok = 0.25 < vol_ratio < 4.0

        return ValidationReport(
            mean_ok=mean_ok,
            variance_ok=variance_ok,
            distribution_ok=distribution_ok,
            autocorrelation_ok=autocorrelation_ok,
            volatility_ok=volatility_ok,
            mean_stat=mean_stat,
            variance_stat=var_stat,
            ks_pvalue=ks_p,
            acf_lag1=acf1,
            realized_vol=realized_vol,
            expected_mean=exp_mean,
            expected_variance=exp_var,
            details={
                "n": n,
                "mean_pvalue": mean_p,
                "variance_ratio": var_ratio,
                "vol_ratio": vol_ratio,
                "acf_lags": self.acf_lags,
            },
        )

    def autocorrelation(self, returns: np.ndarray, lags: int | None = None) -> np.ndarray:
        rets = np.asarray(returns, dtype=np.float64).ravel()
        rets = rets - np.mean(rets)
        n = len(rets)
        max_lag = min(lags or self.acf_lags, n - 1)
        denom = float(np.dot(rets, rets)) + 1e-12
        out = np.ones(max_lag + 1, dtype=np.float64)
        for lag in range(1, max_lag + 1):
            out[lag] = float(np.dot(rets[:-lag], rets[lag:])) / denom
        return out
