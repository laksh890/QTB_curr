"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

``deflated_sharpe_ratio`` implements the Bailey–LdP formula comparing the
observed Sharpe to the expected maximum Sharpe under ``n_trials`` independent
null trials, with a non-normality correction in the sampling variance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

_EULER_MASCHERONI = 0.5772156649015329


def _expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """E[max SR] under null of true SR=0 with i.i.d. Normal returns."""
    n = max(int(n_trials), 1)
    t = max(int(n_obs), 2)
    # Z^{-1}(1 - 1/N) and Z^{-1}(1 - 1/(N e))
    z1 = float(stats.norm.ppf(1.0 - 1.0 / n))
    z2 = float(stats.norm.ppf(1.0 - 1.0 / (n * np.e)))
    if not np.isfinite(z1):
        z1 = 0.0
    if not np.isfinite(z2):
        z2 = z1
    e_max = (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
    return float(e_max / np.sqrt(t - 1))


def sharpe_sampling_se(
    observed_sharpe: float,
    n_obs: int,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Standard error of the Sharpe estimate under non-Normal returns."""
    t = max(int(n_obs), 2)
    sr = float(observed_sharpe)
    g3 = float(skew)
    g4 = float(kurtosis)
    var = (1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr) / (t - 1)
    return float(np.sqrt(max(var, 1e-18)))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    *,
    benchmark_sharpe: float = 0.0,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """PSR = Φ((SR̂ − SR*) / σ̂_SR) — Probabilistic Sharpe Ratio."""
    se = sharpe_sampling_se(observed_sharpe, n_obs, skew=skew, kurtosis=kurtosis)
    z = (float(observed_sharpe) - float(benchmark_sharpe)) / se
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    *,
    annualized: bool = False,
    periods_per_year: float = 252.0,
    return_details: bool = False,
) -> float | dict[str, Any]:
    """Bailey & López de Prado Deflated Sharpe Ratio.

    Parameters
    ----------
    observed_sharpe:
        Non-annualized Sharpe (mean/std of period returns) unless
        ``annualized=True``, in which case it is converted by
        ``SR / sqrt(periods_per_year)``.
    n_trials:
        Number of independent strategy trials / configurations examined.
    n_obs:
        Number of return observations used to estimate the Sharpe.
    skew, kurtosis:
        Sample skewness and kurtosis (Fisher kurtosis + 3, i.e. Normal = 3).

    Returns
    -------
    float
        DSR ∈ (0, 1): probability that the observed Sharpe exceeds the expected
        maximum Sharpe under the multiple-testing null.
    """
    sr = float(observed_sharpe)
    if annualized:
        sr = sr / np.sqrt(max(float(periods_per_year), 1e-12))

    sr0 = _expected_max_sharpe(int(n_trials), int(n_obs))
    se = sharpe_sampling_se(sr, int(n_obs), skew=float(skew), kurtosis=float(kurtosis))
    dsr = float(stats.norm.cdf((sr - sr0) / se))

    if not return_details:
        return dsr
    return {
        "deflated_sharpe": dsr,
        "observed_sharpe": float(sr),
        "benchmark_sharpe_sr0": float(sr0),
        "se": float(se),
        "n_trials": int(n_trials),
        "n_obs": int(n_obs),
        "skew": float(skew),
        "kurtosis": float(kurtosis),
        "probabilistic_sharpe": probabilistic_sharpe_ratio(
            sr, benchmark_sharpe=0.0, n_obs=int(n_obs), skew=float(skew), kurtosis=float(kurtosis)
        ),
    }


# Alias matching common naming
deflated_sharpe = deflated_sharpe_ratio
