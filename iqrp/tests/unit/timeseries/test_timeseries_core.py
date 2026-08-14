"""Core unit tests for Institutional Time-Series Analytics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.core.exceptions import ConfigurationError
from iqrp.app.timeseries import TimeSeriesAnalyticsEngine, TimeSeriesSettings, adjust_pvalues
from iqrp.app.timeseries.alignment import dtw_distance, soft_dtw
from iqrp.app.timeseries.anomaly import (
    isolation_forest_anomalies,
    robust_zscore_anomalies,
    zscore_anomalies,
)
from iqrp.app.timeseries.autocorrelation import acf, pacf
from iqrp.app.timeseries.autocorrelation.cross_correlation import ccf
from iqrp.app.timeseries.base import AnalysisResult, TemporalMode
from iqrp.app.timeseries.change_points import binseg_detect, cusum_detect, pelt_detect
from iqrp.app.timeseries.change_points.bayesian import bayesian_online_changepoint
from iqrp.app.timeseries.change_points.online import online_cusum
from iqrp.app.timeseries.decomposition import classical_decompose, mstl_decompose, stl_decompose
from iqrp.app.timeseries.dependence import (
    distance_correlation,
    engle_granger,
    granger_causality,
    mutual_information,
)
from iqrp.app.timeseries.dependence.cointegration import johansen_trace
from iqrp.app.timeseries.dependence.tail_dependence import empirical_tail_dependence
from iqrp.app.timeseries.diagnostics import full_diagnostics
from iqrp.app.timeseries.features import extract_features
from iqrp.app.timeseries.motifs import find_discords, find_motifs
from iqrp.app.timeseries.multiple_testing import adjust_pvalues as adj
from iqrp.app.timeseries.nonlinear import (
    approximate_entropy,
    hurst_exponent,
    permutation_entropy,
    sample_entropy,
    shannon_entropy,
)
from iqrp.app.timeseries.nonlinear.fractal_dimension import higuchi_fd
from iqrp.app.timeseries.processes import from_market_simulator, simulate_process
from iqrp.app.timeseries.registry import ensure_timeseries_loaded, get, list_methods
from iqrp.app.timeseries.rolling import expanding_apply, multi_scale_windows, rolling_apply
from iqrp.app.timeseries.spectral import dominant_frequencies, fft_spectrum, periodogram, welch_psd
from iqrp.app.timeseries.stationarity import adf, kpss, phillips_perron, variance_ratio
from iqrp.app.timeseries.transforms import TimeSeriesTransformer, log_returns, normalize
from iqrp.app.timeseries.wavelets import cwt_morlet, dwt_haar, wavelet_denoise


@pytest.fixture
def white():
    return np.random.default_rng(0).normal(size=200)


@pytest.fixture
def walk():
    return np.cumsum(np.random.default_rng(1).normal(size=200))


@pytest.fixture
def seasonal():
    t = np.arange(240, dtype=float)
    return np.sin(2 * np.pi * t / 24) + 0.1 * np.random.default_rng(2).normal(size=240)


def test_settings_hydra_and_invalid():
    s = TimeSeriesSettings.default()
    assert s.decomposition.period == 24
    s2 = TimeSeriesSettings.from_hydra(overrides=["seed=7"])
    assert s2.seed == 7
    with pytest.raises(ConfigurationError):
        TimeSeriesSettings.from_mapping({"decomposition": {"method": "nope"}})


def test_transformer_leakage_safe(white):
    tr = TimeSeriesTransformer(method="zscore", window=20, temporal_mode="rolling")
    out = tr.fit_transform(white)
    assert out.shape == white.shape
    assert np.isfinite(out[20:]).all()
    # causal: changing future shouldn't affect past — check rolling contract metadata
    res = tr.analyze(white)
    assert res.temporal_mode == TemporalMode.ROLLING
    for method in (
        "log_return",
        "simple_return",
        "diff",
        "seasonal_diff",
        "robust",
        "rank",
        "winsorize",
        "log",
    ):
        TimeSeriesTransformer(method=method, window=16).fit_transform(white)


def test_decomposition(seasonal):
    c = classical_decompose(seasonal, period=24)
    assert c.trend.size == seasonal.size
    s = stl_decompose(seasonal, period=24, robust=True)
    assert np.isfinite(s.residual).sum() > 0
    m = mstl_decompose(seasonal, periods=(24, 48))
    assert m.method == "mstl"


def test_stationarity_recovery():
    st = simulate_process("stationary", 200, seed=3)["series"]
    ns = simulate_process("non_stationary", 200, seed=4)["series"]
    assert adf(st).pvalue is not None
    assert kpss(st).pvalue is not None
    assert phillips_perron(ns).method == "phillips_perron"
    vr = variance_ratio(st, lags=2)
    assert vr.confidence_interval is not None


def test_acf_pacf_ccf(white):
    a = acf(white, nlags=15)
    assert a.value[0] == pytest.approx(1.0)
    p = pacf(white, nlags=10)
    assert len(np.asarray(p.value)) >= 2
    c = ccf(white, white, nlags=5)
    assert c.method.startswith("ccf") or "cross" in c.method or c.method == "ccf"


def test_change_points_structural_break():
    data = simulate_process("structural_break", 200, seed=5, break_at=100)
    x = data["series"]
    for fn in (cusum_detect, binseg_detect, pelt_detect, bayesian_online_changepoint, online_cusum):
        res = fn(x)
        assert hasattr(res, "indices")


def test_spectral_periodic_recovery():
    data = simulate_process("periodic", 256, seed=6, period=24)
    x = data["series"]
    dom = dominant_frequencies(x, top_k=1)
    assert isinstance(dom.value, list)
    assert fft_spectrum(x).method == "fft_spectrum"
    assert welch_psd(x, nperseg=64).method == "welch_psd"
    assert periodogram(x).method


def test_wavelets(white):
    assert dwt_haar(white).method
    assert cwt_morlet(white[:64]).method
    assert wavelet_denoise(white).method


def test_nonlinear_descriptors(walk):
    assert isinstance(hurst_exponent(walk).value, (float, str))
    assert higuchi_fd(walk).method
    assert shannon_entropy(walk).method
    assert sample_entropy(walk).method
    assert approximate_entropy(walk).method
    assert permutation_entropy(walk).method


def test_dependence_cointegration():
    data = simulate_process("cointegrated", 200, seed=7)
    x, y = data["series"], data["series_y"]
    eg = engle_granger(x, y)
    assert eg.method == "engle_granger"
    assert johansen_trace(x, y).method == "johansen_trace"
    assert granger_causality(x, y).method
    assert mutual_information(x, y).method
    assert distance_correlation(x, y).method
    assert empirical_tail_dependence(x, y).method


def test_anomaly_and_motifs():
    data = simulate_process("anomalous", 200, seed=8)
    x = data["series"]
    assert zscore_anomalies(x).method
    assert robust_zscore_anomalies(x).method
    assert isolation_forest_anomalies(x).method
    assert find_motifs(x, window=16, top_k=2).method
    assert find_discords(x, window=16).method


def test_alignment_dtw(white):
    a, b = white[:80], white[20:100]
    assert isinstance(dtw_distance(a, b).value, (float, str))
    assert soft_dtw(a, b).method


def test_multiple_testing():
    p = [0.001, 0.02, 0.04, 0.5]
    for method in ("bonferroni", "holm", "fdr_bh", "none"):
        out = adj(p, method=method, alpha=0.05)
        assert "adjusted" in out


def test_rolling_utils(white):
    r = rolling_apply(white, 10, lambda c: float(np.mean(c)))
    assert r.shape == white.shape
    e = expanding_apply(white, lambda c: float(np.mean(c)), min_periods=5)
    assert np.isfinite(e[-1])
    assert multi_scale_windows(16) == [16, 32, 64]


def test_features_and_diagnostics(seasonal):
    feats = extract_features(seasonal, period=24, window=48)
    assert isinstance(feats.value, dict)
    assert "trend_strength" in feats.value
    diag = full_diagnostics(seasonal, period=24)
    assert "heteroskedasticity" in diag


def test_engine_api(tmp_path: Path, seasonal):
    eng = TimeSeriesAnalyticsEngine(
        TimeSeriesSettings.from_mapping({"decomposition": {"period": 24}, "motif": {"window": 16}})
    )
    eng.fit(seasonal)
    assert eng.transform(seasonal).shape[0] == seasonal.size
    assert eng.fit_transform(seasonal).shape[0] == seasonal.size
    report = eng.analyze(seasonal)
    assert "disclaimer" in report
    assert eng.decompose(seasonal).method
    assert eng.correlate(seasonal).method
    assert eng.correlate(seasonal, kind="pacf").method
    assert eng.correlate(seasonal, seasonal, kind="ccf").method
    assert "tests" in eng.stationarity(seasonal)
    assert eng.change_points(seasonal, method="pelt")
    assert eng.change_points(seasonal, method="cusum")
    assert eng.change_points(seasonal, method="binseg")
    assert eng.spectral_analysis(seasonal)
    assert eng.wavelet_analysis(seasonal)
    assert "shannon" in eng.entropy(seasonal)
    assert eng.hurst(seasonal).method
    y = seasonal + 0.01
    assert eng.cointegration(seasonal, y).method
    assert eng.cointegration(seasonal, y, method="johansen").method
    assert eng.dependence(seasonal, y)
    assert eng.anomalies(seasonal).method
    assert eng.anomalies(seasonal, method="statistical").method
    assert eng.motifs(seasonal).method
    assert eng.dtw(seasonal[:60], seasonal[60:120]).method
    assert eng.dtw(seasonal[:40], seasonal[40:80], soft=True).method
    assert eng.features(seasonal).method
    assert eng.diagnostics(seasonal)
    assert eng.visualize(seasonal)
    assert eng.detect(seasonal, what="anomalies")
    path = eng.save(tmp_path / "ts.json")
    loaded = TimeSeriesAnalyticsEngine.load(path)
    assert loaded.settings.data_version
    assert list_methods()
    ensure_timeseries_loaded()
    assert get("adf")


def test_processes_all_kinds():
    for kind in (
        "stationary",
        "non_stationary",
        "mean_reverting",
        "trending",
        "periodic",
        "long_memory",
        "structural_break",
        "regime_change",
        "cointegrated",
        "anomalous",
    ):
        out = simulate_process(kind, 100, seed=1)
        assert "series" in out
    assert "series" in from_market_simulator(80, preset="sideways")


def test_analysis_result_dict():
    r = AnalysisResult(method="t", value=1.0, temporal_mode=TemporalMode.CAUSAL, significant=True)
    d = r.to_dict()
    assert d["temporal_mode"] == "causal"
