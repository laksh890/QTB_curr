"""Hypothesis testing suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from iqrp.app.math._array import as_vector


@dataclass(frozen=True, slots=True)
class TestResult:
    statistic: float
    pvalue: float
    name: str
    df: float | None = None
    alternative: str = "two-sided"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ttest_1samp(x: Any, popmean: float = 0.0) -> TestResult:
    r = stats.ttest_1samp(as_vector(x), popmean=popmean)
    return TestResult(float(r.statistic), float(r.pvalue), "ttest_1samp", df=float(r.df))


def ttest_ind(x: Any, y: Any, *, equal_var: bool = True) -> TestResult:
    r = stats.ttest_ind(as_vector(x), as_vector(y), equal_var=equal_var)
    name = "welch_ttest" if not equal_var else "ttest_ind"
    return TestResult(float(r.statistic), float(r.pvalue), name, df=float(getattr(r, "df", np.nan)))


def anova(*groups: Any) -> TestResult:
    arrays = [as_vector(g) for g in groups]
    r = stats.f_oneway(*arrays)
    return TestResult(float(r.statistic), float(r.pvalue), "anova")


def chi_square_test(observed: Any, expected: Any | None = None) -> TestResult:
    obs = as_array_int(observed)
    if expected is None:
        r = stats.chisquare(obs)
    else:
        r = stats.chisquare(obs, f_exp=as_vector(expected))
    return TestResult(float(r.statistic), float(r.pvalue), "chi_square")


def ks_test(x: Any, cdf: str = "norm", **params: Any) -> TestResult:
    r = stats.kstest(as_vector(x), cdf, args=tuple(params.values()) if params else ())
    return TestResult(float(r.statistic), float(r.pvalue), "kolmogorov_smirnov")


def mann_whitney(x: Any, y: Any) -> TestResult:
    r = stats.mannwhitneyu(as_vector(x), as_vector(y), alternative="two-sided")
    return TestResult(float(r.statistic), float(r.pvalue), "mann_whitney")


def wilcoxon(x: Any, y: Any | None = None) -> TestResult:
    r = stats.wilcoxon(as_vector(x)) if y is None else stats.wilcoxon(as_vector(x), as_vector(y))
    return TestResult(float(r.statistic), float(r.pvalue), "wilcoxon")


def shapiro_wilk(x: Any) -> TestResult:
    r = stats.shapiro(as_vector(x))
    return TestResult(float(r.statistic), float(r.pvalue), "shapiro_wilk")


def jarque_bera(x: Any) -> TestResult:
    r = stats.jarque_bera(as_vector(x))
    return TestResult(float(r.statistic), float(r.pvalue), "jarque_bera")


def adf_test(x: Any, *, max_lag: int | None = None) -> TestResult:
    """Augmented Dickey-Fuller (constant only), pure NumPy/SciPy implementation."""
    y = as_vector(x).astype(np.float64)
    n = len(y)
    if n < 10:
        return TestResult(float("nan"), float("nan"), "adf")
    dy = np.diff(y)
    y_lag = y[:-1]
    lag = max_lag if max_lag is not None else int(np.ceil(12 * (n / 100) ** 0.25))
    lag = int(np.clip(lag, 0, max(0, len(dy) // 3)))
    # Build regression: dy_t = a + b y_{t-1} + sum c_i dy_{t-i}
    rows = []
    target = []
    for t in range(lag, len(dy)):
        row = [1.0, y_lag[t]]
        for i in range(1, lag + 1):
            row.append(dy[t - i])
        rows.append(row)
        target.append(dy[t])
    design = np.asarray(rows, dtype=np.float64)
    response = np.asarray(target, dtype=np.float64)
    beta, *_ = np.linalg.lstsq(design, response, rcond=None)
    resid = response - design @ beta
    s2 = float(np.dot(resid, resid) / max(len(response) - design.shape[1], 1))
    xtx_inv = np.linalg.pinv(design.T @ design)
    se_b = float(np.sqrt(max(s2 * xtx_inv[1, 1], 0.0)))
    stat = float(beta[1] / se_b) if se_b > 0 else float("nan")
    # Approximate p-value via MacKinnon-like logistic (constant case)
    p = _mackinnon_pvalue(stat)
    return TestResult(stat, p, "adf", df=float(lag))


def kpss_test(x: Any, *, regression: str = "c") -> TestResult:
    """KPSS stationarity test (level), pure NumPy implementation."""
    y = as_vector(x).astype(np.float64)
    n = len(y)
    if n < 10:
        return TestResult(float("nan"), float("nan"), "kpss")
    if regression == "ct":
        t = np.arange(1, n + 1, dtype=np.float64)
        design = np.column_stack([np.ones(n), t])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
    else:
        resid = y - np.mean(y)
    s = np.cumsum(resid)
    lags = int(np.floor(4 * (n / 100) ** 0.25))
    # Newey-West long-run variance
    gamma0 = float(np.dot(resid, resid) / n)
    lrv = gamma0
    for h in range(1, lags + 1):
        w = 1.0 - h / (lags + 1)
        gamma = float(np.dot(resid[h:], resid[:-h]) / n)
        lrv += 2.0 * w * gamma
    lrv = max(lrv, 1e-15)
    stat = float(np.sum(s**2) / (n**2 * lrv))
    # Critical values for level: 10%=0.347, 5%=0.463, 1%=0.739
    p = _kpss_pvalue(stat)
    return TestResult(stat, p, "kpss")


def as_array_int(x: Any) -> np.ndarray:
    return np.asarray(as_vector(x), dtype=np.float64)


def _mackinnon_pvalue(stat: float) -> float:
    # Rough approximation for constant ADF
    # Critical approx: 1%=-3.43, 5%=-2.86, 10%=-2.57
    if not np.isfinite(stat):
        return float("nan")
    if stat <= -3.43:
        return 0.01
    if stat <= -2.86:
        return 0.03
    if stat <= -2.57:
        return 0.08
    if stat <= -1.95:
        return 0.2
    return min(0.99, 0.5 + 0.1 * (stat + 1.95))


def _kpss_pvalue(stat: float) -> float:
    if not np.isfinite(stat):
        return float("nan")
    if stat >= 0.739:
        return 0.01
    if stat >= 0.463:
        return 0.03
    if stat >= 0.347:
        return 0.08
    return min(0.99, 0.5)
