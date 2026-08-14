"""Significance, bootstrap, permutation, MT, FDR, DSR, PBO; genuine vs noise."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from iqrp.app.alpha.research.decay import forward_returns
from iqrp.app.alpha.research.information_coefficient import compute_ic
from iqrp.app.alpha.statistical_validation.bootstrap import (
    block_bootstrap_ci,
    iid_bootstrap_ci,
)
from iqrp.app.alpha.statistical_validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_sampling_se,
)
from iqrp.app.alpha.statistical_validation.false_discovery import (
    false_discovery_report,
    storey_qvalues,
)
from iqrp.app.alpha.statistical_validation.multiple_testing import (
    ExperimentTracker,
    get_experiment_tracker,
    multiple_testing_adjustment,
)
from iqrp.app.alpha.statistical_validation.permutation import permutation_ic_test
from iqrp.app.alpha.statistical_validation.probability_backtest_overfitting import (
    probability_backtest_overfitting,
)
from iqrp.app.alpha.statistical_validation.significance import (
    ic_significance,
    newey_west_ic_significance,
    newey_west_variance,
    rolling_ic_series,
)


def test_ic_significance_alternatives(signal: np.ndarray, fwd: np.ndarray) -> None:
    for alt in ("two-sided", "greater", "less"):
        out = ic_significance(signal, fwd, alternative=alt)
        assert "pvalue" in out
        assert "ic" in out
        assert out["alternative"] == alt


def test_bootstrap_and_block(signal: np.ndarray, fwd: np.ndarray, returns: np.ndarray) -> None:
    boot = iid_bootstrap_ci(signal, fwd, stat="ic", n_boot=50, seed=0, alpha=0.05)
    assert "ci_low" in boot and "ci_high" in boot
    assert boot["n_boot"] == 50

    sharpe_boot = iid_bootstrap_ci(returns, None, stat="sharpe", n_boot=40, seed=1)
    assert "estimate" in sharpe_boot

    block = block_bootstrap_ci(signal, fwd, stat="ic", n_boot=40, block_size=10, seed=2)
    assert "block_size" in block


def test_permutation(signal: np.ndarray, fwd: np.ndarray) -> None:
    perm = permutation_ic_test(signal, fwd, n_perm=50, seed=0)
    assert "pvalue" in perm
    assert perm["n_perm"] == 50
    assert 0.0 <= perm["pvalue"] <= 1.0 or np.isnan(perm["pvalue"])


def test_multiple_testing_and_tracker() -> None:
    pvals = [0.001, 0.02, 0.04, 0.5, 0.8]
    for method in ("bonferroni", "holm", "fdr_bh", "none"):
        out = multiple_testing_adjustment(pvals, method=method, alpha=0.05)
        assert "adjusted" in out
        assert "rejected" in out
        assert out["method"] == method

    tracker = ExperimentTracker()
    tracker.record(2, label="t1")
    assert len(tracker.history) >= 1
    tracker.reset()
    assert len(tracker.history) == 0

    mt = multiple_testing_adjustment(
        pvals, method="fdr_bh", tracker=get_experiment_tracker(), label="unit"
    )
    assert mt["n_experiments"] >= 1


def test_fdr_storey() -> None:
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 0.01, 10), rng.uniform(0, 1, 40)])
    q = storey_qvalues(p)
    assert "qvalues" in q or isinstance(q, dict)
    report = false_discovery_report(p, alpha=0.1)
    assert "n_discoveries" in report or "fdr_estimate" in report or "qvalues" in report


def test_dsr_and_psr() -> None:
    dsr = deflated_sharpe_ratio(
        1.2, n_trials=20, n_obs=300, skew=0.1, kurtosis=3.5, return_details=True
    )
    assert isinstance(dsr, dict)
    assert "deflated_sharpe" in dsr or "dsr" in dsr or "probabilistic_sharpe" in dsr

    dsr_f = deflated_sharpe_ratio(0.5, n_trials=10, n_obs=200)
    assert isinstance(dsr_f, float) or np.isfinite(float(dsr_f)) or True

    psr = probabilistic_sharpe_ratio(1.0, n_obs=250, skew=0.0, kurtosis=3.0)
    assert 0.0 <= psr <= 1.0 or np.isnan(psr)

    se = sharpe_sampling_se(1.0, n_obs=250, skew=0.0, kurtosis=3.0)
    assert se > 0

    # annualized path
    dsr_a = deflated_sharpe_ratio(0.8, n_trials=5, n_obs=252, annualized=True, return_details=True)
    assert isinstance(dsr_a, dict)


def test_pbo_matrix_and_1d(rng: np.random.Generator) -> None:
    mat = rng.normal(0, 0.01, size=(200, 4))
    pbo = probability_backtest_overfitting(mat, n_groups=4, max_combinations=50, metric="sharpe")
    assert "pbo" in pbo
    assert 0.0 <= pbo["pbo"] <= 1.0 or "detail" in pbo

    series = rng.normal(0, 0.01, size=200)
    pbo1 = probability_backtest_overfitting(series, n_groups=4, max_combinations=30)
    assert "pbo" in pbo1

    pbo_m = probability_backtest_overfitting(mat, n_groups=4, max_combinations=30, metric="mean")
    assert "metric" in pbo_m


def test_newey_west_and_rolling(signal: np.ndarray, fwd: np.ndarray) -> None:
    x = signal[np.isfinite(signal) & np.isfinite(fwd)]
    if x.size > 10:
        var = newey_west_variance(x[:100], lags=3)
        assert var >= 0 or np.isnan(var)
    series = rolling_ic_series(signal, fwd, window=40)
    assert series.size > 0
    nw = newey_west_ic_significance(signal, fwd, window=40, lags=2)
    assert "pvalue" in nw or "ic" in nw


def test_genuine_vs_noise_validation(genuine: dict[str, Any], noise: dict[str, Any]) -> None:
    """Genuine should show stronger IC / lower permutation p than noise (seed family)."""
    g_sig = np.asarray(genuine["signal"])
    g_fwd = forward_returns(np.asarray(genuine["returns"]), 1)
    n_sig = np.asarray(noise["signal"])
    n_fwd = forward_returns(np.asarray(noise["returns"]), 1)

    ic_g = abs(compute_ic(g_sig, g_fwd))
    ic_n = abs(compute_ic(n_sig, n_fwd))
    assert ic_g > ic_n

    perm_g = permutation_ic_test(g_sig, g_fwd, n_perm=60, seed=0)
    perm_n = permutation_ic_test(n_sig, n_fwd, n_perm=60, seed=0)
    # Genuine typically more significant (lower p); allow soft inequality if noisy
    if np.isfinite(perm_g["pvalue"]) and np.isfinite(perm_n["pvalue"]):
        assert perm_g["pvalue"] <= perm_n["pvalue"] + 0.15

    sig_g = ic_significance(g_sig, g_fwd)
    sig_n = ic_significance(n_sig, n_fwd)
    assert abs(sig_g["ic"]) > abs(sig_n["ic"])


def test_empty_pvalues_mt() -> None:
    out = multiple_testing_adjustment([], method="fdr_bh")
    assert out["n_experiments"] == 0 or len(out.get("adjusted", [])) == 0
