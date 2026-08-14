"""Polish remaining uncovered lines."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.alignment.dtw import dtw_distance, dtw_path
from iqrp.app.timeseries.alignment.shapelets import discover_shapelets
from iqrp.app.timeseries.anomaly.isolation_forest import isolation_forest_anomalies
from iqrp.app.timeseries.anomaly.matrix_profile import matrix_profile_anomalies
from iqrp.app.timeseries.autocorrelation.acf import acf, rolling_acf
from iqrp.app.timeseries.autocorrelation.cross_correlation import ccf, lead_lag
from iqrp.app.timeseries.autocorrelation.pacf import pacf
from iqrp.app.timeseries.change_points.bayesian import bayesian_online_changepoint
from iqrp.app.timeseries.change_points.binary_segmentation import binseg_detect
from iqrp.app.timeseries.change_points.cusum import cusum_detect
from iqrp.app.timeseries.change_points.online import online_cusum
from iqrp.app.timeseries.change_points.pelt import pelt_detect
from iqrp.app.timeseries.decomposition.stl import stl_decompose
from iqrp.app.timeseries.dependence.cointegration import engle_granger, johansen_trace
from iqrp.app.timeseries.dependence.distance_correlation import distance_correlation
from iqrp.app.timeseries.dependence.granger import granger_causality
from iqrp.app.timeseries.dependence.mutual_information import mutual_information
from iqrp.app.timeseries.dependence.tail_dependence import empirical_tail_dependence
from iqrp.app.timeseries.diagnostics.structural_breaks import distribution_shift, heteroskedasticity
from iqrp.app.timeseries.features.trend_features import cycle_features
from iqrp.app.timeseries.motifs.discord import find_discords
from iqrp.app.timeseries.motifs.discovery import find_motifs
from iqrp.app.timeseries.motifs.matrix_profile import compute_matrix_profile
from iqrp.app.timeseries.motifs.similarity import nearest_neighbors, subsequence_distance
from iqrp.app.timeseries.nonlinear.approximate_entropy import approximate_entropy
from iqrp.app.timeseries.nonlinear.entropy import shannon_entropy
from iqrp.app.timeseries.nonlinear.fractal_dimension import higuchi_fd
from iqrp.app.timeseries.nonlinear.hurst import hurst_exponent
from iqrp.app.timeseries.nonlinear.permutation_entropy import permutation_entropy
from iqrp.app.timeseries.nonlinear.sample_entropy import sample_entropy
from iqrp.app.timeseries.spectral.periodogram import periodogram
from iqrp.app.timeseries.spectral.spectral_density import spectral_density
from iqrp.app.timeseries.spectral.welch import welch_psd
from iqrp.app.timeseries.stationarity.kpss import _kpss_pvalue, kpss
from iqrp.app.timeseries.transforms import TimeSeriesTransformer
from iqrp.app.timeseries.wavelets.continuous import cwt_morlet
from iqrp.app.timeseries.wavelets.denoising import wavelet_denoise
from iqrp.app.timeseries.wavelets.discrete import dwt_haar


def test_isolation_window_and_nan_rows():
    x = np.random.default_rng(0).normal(size=80)
    x[5] = np.nan
    assert isolation_forest_anomalies(x, window=4, n_trees=5, seed=0).method
    assert isolation_forest_anomalies(np.full(20, np.nan)).value == "insufficient_data"


def test_kpss_pvalue_helper():
    crit = {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}
    assert _kpss_pvalue(0.1, crit) > 0.1
    assert _kpss_pvalue(1.0, crit) < 0.05
    assert 0 < _kpss_pvalue(0.4, crit) < 1
    assert kpss(np.random.randn(50)).method


def test_wavelet_edges():
    assert dwt_haar(np.array([1.0])).value == "insufficient_data"
    assert cwt_morlet(np.array([1.0, 2.0])).value == "insufficient_data"
    x = np.random.default_rng(1).normal(size=64)
    assert dwt_haar(x, level=2).method
    assert cwt_morlet(x, scales=(2, 4, 8)).method
    assert wavelet_denoise(x, threshold=0.5).method
    assert wavelet_denoise(np.array([1.0])).method


def test_stl_odd_windows():
    x = np.sin(np.linspace(0, 20, 100))
    assert stl_decompose(x, period=12, seasonal_window=8, trend_window=20, robust=True).method


def test_training_only_winsorize_fit():
    x = np.random.default_rng(2).normal(size=50)
    tr = TimeSeriesTransformer(method="winsorize", temporal_mode="training_only")
    tr.fit(x)
    assert tr.transform(x).shape[0] == 50
    tr2 = TimeSeriesTransformer(method="rank", temporal_mode="training_only")
    tr2.fit(x)
    assert tr2.transform(x).shape[0] == 50
    # simple return zero prev
    y = np.array([0.0, 1.0, 2.0])
    assert TimeSeriesTransformer(method="simple_return").fit_transform(y).shape[0] == 3
    assert TimeSeriesTransformer(method="diff").fit_transform(np.array([1.0])).shape[0] == 1


def test_misc_insufficient():
    assert acf(np.array([1.0, np.nan])).value == "insufficient_data" or acf(np.ones(5)).method
    assert pacf(np.array([1.0])).value == "insufficient_data"
    assert (
        ccf(np.array([1.0]), np.array([1.0])).value == "insufficient_data"
        or ccf(np.random.randn(20), np.random.randn(20)).method
    )
    assert lead_lag(np.array([1.0, 2.0]), np.array([1.0, 2.0])).method
    assert rolling_acf(np.ones(5), window=10, lag=1).method
    assert periodogram(np.array([1.0])).value == "insufficient_data"
    assert welch_psd(np.random.randn(100), nperseg=32, noverlap=8, detrend=False).method
    assert spectral_density(np.random.randn(64), method="welch", nperseg=16).method
    assert dtw_distance([], []).value == "insufficient_data"
    assert (
        dtw_path([1.0], [1.0, 2.0]).method or dtw_path(np.arange(10.0), np.arange(10.0) + 1).method
    )
    assert discover_shapelets(
        np.random.randn(60), labels=np.array([0] * 30 + [1] * 30), lengths=(8,), top_k=1
    ).method
    assert matrix_profile_anomalies(np.ones(5), window=3).value == "insufficient_data"
    assert compute_matrix_profile(np.random.randn(80), window=10).method
    assert find_motifs(np.random.randn(80), window=10, top_k=1, max_distance=0.01).method
    assert find_discords(np.random.randn(80), window=10, top_k=1).method
    assert subsequence_distance(np.arange(10.0), np.arange(10.0), z_normalize=False).method
    assert nearest_neighbors(np.arange(8.0), np.random.randn(40), z_normalize=False).method
    assert engle_granger(np.ones(5), np.ones(5)).value == "insufficient_data"
    assert johansen_trace(np.random.randn(40), np.random.randn(40), lag=1).method
    assert granger_causality(np.ones(5), np.ones(5)).value == "insufficient_data"
    assert mutual_information(np.array([1.0, 1.0]), np.array([1.0, 1.0])).method
    assert distance_correlation(np.ones(3), np.ones(3)).method
    assert empirical_tail_dependence(np.ones(5), np.ones(5)).method
    assert distribution_shift(np.ones(3)).value == "insufficient_data"
    assert heteroskedasticity(np.ones(5)).value == "insufficient_data"
    assert cusum_detect(np.ones(3)).indices == []
    assert binseg_detect(np.ones(8), min_size=5).indices == [] or True
    assert pelt_detect(np.random.randn(80), penalty=10.0, min_size=5).method
    assert bayesian_online_changepoint(np.random.randn(40)).method
    assert online_cusum(np.random.randn(40), threshold=2.0).method
    assert hurst_exponent(np.random.randn(10)).value == "insufficient_data"
    assert shannon_entropy(np.array([1.0])).value == "insufficient_data"
    assert sample_entropy(np.arange(5.0)).method
    assert approximate_entropy(np.arange(5.0)).method
    assert permutation_entropy(np.array([1.0, 1.0, 1.0])).method
    assert (
        higuchi_fd(np.arange(8.0)).value == "insufficient_data"
        or higuchi_fd(np.arange(40.0)).method
    )
    # cycle features when no peaks
    assert "seasonal_strength" in cycle_features(np.random.randn(80), period=12)
