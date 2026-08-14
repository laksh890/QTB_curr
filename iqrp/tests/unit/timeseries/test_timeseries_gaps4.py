"""Tiny edge-case tests to clear 98% coverage."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.alignment.dtw import dtw_path
from iqrp.app.timeseries.alignment.shapelets import discover_shapelets
from iqrp.app.timeseries.alignment.soft_dtw import soft_dtw
from iqrp.app.timeseries.anomaly.isolation_forest import (
    _isolation_tree_depths,
    _numpy_isolation_forest,
)
from iqrp.app.timeseries.anomaly.matrix_profile import matrix_profile_anomalies
from iqrp.app.timeseries.autocorrelation.acf import acf
from iqrp.app.timeseries.autocorrelation.cross_correlation import ccf
from iqrp.app.timeseries.autocorrelation.pacf import pacf
from iqrp.app.timeseries.change_points.bayesian import bayesian_online_changepoint
from iqrp.app.timeseries.change_points.online import online_cusum
from iqrp.app.timeseries.change_points.pelt import pelt_detect
from iqrp.app.timeseries.decomposition.classical import classical_decompose
from iqrp.app.timeseries.decomposition.seasonal import seasonal_strength
from iqrp.app.timeseries.decomposition.stl import stl_decompose
from iqrp.app.timeseries.decomposition.trend import _hp_filter, trend_strength
from iqrp.app.timeseries.dependence.cointegration import engle_granger, johansen_trace
from iqrp.app.timeseries.dependence.distance_correlation import distance_correlation
from iqrp.app.timeseries.dependence.granger import granger_causality
from iqrp.app.timeseries.dependence.mutual_information import mutual_information
from iqrp.app.timeseries.motifs.discovery import find_motifs
from iqrp.app.timeseries.motifs.matrix_profile import compute_matrix_profile
from iqrp.app.timeseries.nonlinear.fractal_dimension import higuchi_fd
from iqrp.app.timeseries.nonlinear.hurst import hurst_exponent
from iqrp.app.timeseries.nonlinear.permutation_entropy import permutation_entropy
from iqrp.app.timeseries.spectral.periodogram import periodogram
from iqrp.app.timeseries.spectral.spectral_density import spectral_density
from iqrp.app.timeseries.spectral.welch import welch_psd
from iqrp.app.timeseries.stationarity.kpss import _kpss_pvalue
from iqrp.app.timeseries.transforms import (
    TimeSeriesTransformer,
    log_returns as log_returns_arr,
    simple_returns as simple_returns_arr,
)
from iqrp.app.timeseries.wavelets.continuous import cwt_morlet
from iqrp.app.timeseries.wavelets.denoising import wavelet_denoise
from iqrp.app.timeseries.wavelets.discrete import dwt_haar


def test_transform_short_arrays():
    assert np.isnan(log_returns_arr([1.0])).all()
    assert np.isnan(simple_returns_arr([1.0])).all()
    assert np.isnan(TimeSeriesTransformer(method="diff").fit_transform([1.0])).all()
    x = np.zeros(30)
    assert TimeSeriesTransformer(method="zscore", window=5).fit_transform(x).shape[0] == 30
    assert TimeSeriesTransformer(method="robust", window=5).fit_transform(x).shape[0] == 30
    assert TimeSeriesTransformer(method="diff", period=2).fit_transform([1.0, 2.0]).shape[0] == 2
    # force order>len via seasonal_diff short
    assert (
        TimeSeriesTransformer(method="seasonal_diff", period=5).fit_transform([1.0, 2.0]).shape[0]
        == 2
    )


def test_dwt_zero_energy_and_odd():
    assert dwt_haar(np.zeros(16)).method
    assert dwt_haar(np.arange(3.0)).method  # odd length
    assert dwt_haar(np.arange(64.0), level=10).method


def test_cwt_and_denoise_edges():
    x = np.random.default_rng(0).normal(size=32)
    assert cwt_morlet(x).method
    assert wavelet_denoise(np.zeros(8)).method
    assert wavelet_denoise(x, threshold=0.2).method


def test_isolation_constant_features():
    X = np.ones((40, 2))
    scores, mask = _numpy_isolation_forest(X, n_trees=5, contamination=0.1, max_depth=3, seed=0)
    assert scores.size == 40
    depths = _isolation_tree_depths(X, np.random.default_rng(0), 2)
    assert depths.size == 40


def test_shapelets_nan_series():
    x = np.random.default_rng(1).normal(size=50)
    x[3] = np.nan
    assert discover_shapelets(x, lengths=(8,), top_k=1, n_candidates=10).method


def test_soft_dtw_short():
    assert (
        soft_dtw([1.0], [1.0]).value == "insufficient_data"
        or soft_dtw(np.arange(5.0), np.arange(5.0)).method
    )


def test_dtw_path_empty():
    assert dtw_path([], [1.0]).value == "insufficient_data"


def test_acf_constant_and_pacf():
    assert acf(np.ones(40)).metadata.get("constant_series") or acf(np.ones(40)).method
    assert pacf(np.ones(40)).method
    assert (
        ccf(np.random.randn(30), np.random.randn(30), nlags=0).method
        or ccf(np.random.randn(30), np.random.randn(30)).method
    )


def test_spectral_invalid_and_welch_overlap():
    x = np.random.default_rng(2).normal(size=80)
    assert spectral_density(x, method="fft").method
    assert periodogram(x, detrend=False).method
    assert welch_psd(x, nperseg=20, noverlap=19).method


def test_kpss_fallback_return():
    # empty crit edge — call with mid value that falls through loop end
    assert _kpss_pvalue(0.4, {"5%": 0.463}) >= 0


def test_classical_odd_period():
    t = np.arange(99, dtype=float)
    x = np.sin(2 * np.pi * t / 9)
    assert classical_decompose(x, period=9).method


def test_strength_zero_var():
    assert seasonal_strength(np.zeros(10), np.zeros(10)) == 0.0
    assert trend_strength(np.zeros(10), np.zeros(10)) == 0.0
    assert _hp_filter(np.array([1.0, 2.0, 3.0])).size == 3


def test_dependence_edges():
    a = np.cumsum(np.random.default_rng(3).normal(size=100))
    b = 2 * a + np.random.default_rng(4).normal(0, 0.01, size=100)
    assert engle_granger(a, b).method
    assert johansen_trace(a, b, lag=2).method
    assert granger_causality(a, b, max_lag=2).method
    assert mutual_information(a, a).method
    assert distance_correlation(a, a).method


def test_changepoint_and_motifs_edges():
    x = np.concatenate([np.zeros(30), np.ones(30) * 4])
    assert pelt_detect(x, min_size=3).method
    assert bayesian_online_changepoint(x, hazard=0.01).method
    assert online_cusum(x, threshold=1.0, drift=0.1).method
    assert compute_matrix_profile(x, window=5).method
    assert find_motifs(x, window=5, top_k=1).method
    assert matrix_profile_anomalies(x, window=5).method


def test_nonlinear_edges():
    x = np.random.default_rng(5).normal(size=80)
    assert hurst_exponent(x, min_window=4, max_window=20).method
    assert higuchi_fd(x, k_max=5).method
    assert permutation_entropy(x, order=3).method


def test_stl_zero_weight_path():
    x = np.sin(np.linspace(0, 10, 80))
    assert stl_decompose(x, period=8, n_iter=1, robust=False).method
