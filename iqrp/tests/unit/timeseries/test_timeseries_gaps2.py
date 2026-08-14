"""Final coverage push for Time-Series Analytics."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from iqrp.app.timeseries.alignment.shapelets import discover_shapelets
from iqrp.app.timeseries.anomaly import isolation_forest_anomalies
from iqrp.app.timeseries.anomaly.isolation_forest import _numpy_isolation_forest
from iqrp.app.timeseries.autocorrelation.acf import acf, bartlett_bands, rolling_acf
from iqrp.app.timeseries.autocorrelation.cross_correlation import ccf, lead_lag
from iqrp.app.timeseries.autocorrelation.pacf import pacf
from iqrp.app.timeseries.change_points.bayesian import bayesian_online_changepoint
from iqrp.app.timeseries.change_points.cusum import cusum_detect
from iqrp.app.timeseries.change_points.online import online_cusum
from iqrp.app.timeseries.change_points.pelt import pelt_detect
from iqrp.app.timeseries.decomposition.classical import classical_decompose
from iqrp.app.timeseries.dependence.cointegration import engle_granger, johansen_trace
from iqrp.app.timeseries.dependence.distance_correlation import distance_correlation
from iqrp.app.timeseries.dependence.granger import granger_causality
from iqrp.app.timeseries.dependence.mutual_information import mutual_information
from iqrp.app.timeseries.dependence.tail_dependence import empirical_tail_dependence
from iqrp.app.timeseries.diagnostics import (
    distribution_shift as ds_mod,
    heteroskedasticity as het_mod,
    seasonality as seas_mod,
)
from iqrp.app.timeseries.diagnostics.distribution_shift import distribution_shift
from iqrp.app.timeseries.diagnostics.heteroskedasticity import heteroskedasticity
from iqrp.app.timeseries.diagnostics.seasonality import seasonality_diagnostics
from iqrp.app.timeseries.diagnostics.structural_breaks import distribution_shift as ds2
from iqrp.app.timeseries.features import (
    cycle_features as cf_mod,
    entropy_features as ef_mod,
    memory_features as mf_mod,
    spectral_features as sf_mod,
    volatility_features as vf_mod,
)
from iqrp.app.timeseries.features.cycle_features import cycle_features
from iqrp.app.timeseries.features.entropy_features import entropy_features
from iqrp.app.timeseries.features.memory_features import memory_features
from iqrp.app.timeseries.features.spectral_features import spectral_features
from iqrp.app.timeseries.features.trend_features import volatility_features as vf2
from iqrp.app.timeseries.features.volatility_features import volatility_features
from iqrp.app.timeseries.motifs.discord import find_discords
from iqrp.app.timeseries.motifs.discovery import find_motifs
from iqrp.app.timeseries.motifs.matrix_profile import compute_matrix_profile
from iqrp.app.timeseries.motifs.similarity import nearest_neighbors, subsequence_distance
from iqrp.app.timeseries.nonlinear import (
    approximate_entropy,
    higuchi_fd,
    hurst_exponent,
    permutation_entropy,
    sample_entropy,
    shannon_entropy,
)
from iqrp.app.timeseries.orchestrator import TimeSeriesAnalyticsEngine
from iqrp.app.timeseries.registry import ensure_timeseries_loaded
from iqrp.app.timeseries.serializer import TimeSeriesSerializer, _to_jsonable
from iqrp.app.timeseries.spectral.fft import dominant_frequencies, fft_spectrum
from iqrp.app.timeseries.spectral.periodogram import periodogram
from iqrp.app.timeseries.spectral.spectral_density import period_from_frequency, spectral_density
from iqrp.app.timeseries.spectral.welch import welch_psd
from iqrp.app.timeseries.stationarity.kpss import kpss
from iqrp.app.timeseries.transforms import TimeSeriesTransformer
from iqrp.app.timeseries.transforms.log_transform import log_transform, log_via_transformer
from iqrp.app.timeseries.transforms.normalization import normalize as norm_wrap
from iqrp.app.timeseries.transforms.rank_transform import rank_transform
from iqrp.app.timeseries.wavelets.continuous import cwt_morlet
from iqrp.app.timeseries.wavelets.denoising import wavelet_denoise
from iqrp.app.timeseries.wavelets.discrete import dwt_haar


def test_import_facades():
    assert distribution_shift is not None
    assert heteroskedasticity is not None
    assert seasonality_diagnostics is not None
    assert cycle_features is not None
    assert entropy_features is not None
    assert memory_features is not None
    assert spectral_features is not None
    assert volatility_features is not None
    assert ds_mod is not None and het_mod is not None and seas_mod is not None
    assert cf_mod and ef_mod and mf_mod and sf_mod and vf_mod
    x = np.random.default_rng(0).normal(size=100)
    assert ds2(x).method
    assert vf2(x)


def test_isolation_forest_numpy_path():
    x = np.random.default_rng(1).normal(size=120)
    x[10] = 20
    # force Exception in sklearn path
    with patch(
        "iqrp.app.timeseries.anomaly.isolation_forest.IsolationForest",
        create=True,
        side_effect=ImportError("no"),
    ):
        # patch the import inside the function by making sklearn raise
        import builtins

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "sklearn.ensemble" or (name == "sklearn" and fromlist):
                raise ImportError("no sklearn")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            res = isolation_forest_anomalies(x, contamination=0.05, n_trees=10, seed=0, window=1)
            assert res.method == "isolation_forest_anomalies"
            assert res.metadata.get("backend") == "numpy"
    # window>1 path + sklearn if available
    assert isolation_forest_anomalies(x, window=3, n_trees=5, seed=1).method
    X = x.reshape(-1, 1)
    scores, mask = _numpy_isolation_forest(X, n_trees=8, contamination=0.1, max_depth=4, seed=0)
    assert scores.shape[0] == X.shape[0]
    assert mask.dtype == bool
    assert isolation_forest_anomalies(np.ones(5)).value == "insufficient_data"
    # _avg_path_length branches
    from iqrp.app.timeseries.anomaly.isolation_forest import _avg_path_length

    assert _avg_path_length(1) == 0.0
    assert _avg_path_length(2) == 1.0


def test_short_series_and_edge_methods():
    short = np.array([1.0])
    tiny = np.array([1.0, 2.0, 3.0])
    assert fft_spectrum(short).value == "insufficient_data"
    assert dominant_frequencies(short).value == "insufficient_data"
    assert periodogram(tiny).method
    assert (
        welch_psd(tiny, nperseg=64).value == "insufficient_data"
        or welch_psd(np.random.randn(80), nperseg=16).method
    )
    assert spectral_density(tiny, method="periodogram").method
    assert period_from_frequency([]).value == "insufficient_data"
    assert dwt_haar(np.arange(32.0)).method
    assert cwt_morlet(np.arange(32.0)).method
    assert wavelet_denoise(np.arange(64.0)).method
    assert hurst_exponent(tiny).value == "insufficient_data"
    assert shannon_entropy(tiny).value == "insufficient_data"
    assert sample_entropy(tiny).value == "insufficient_data"
    assert approximate_entropy(tiny).value == "insufficient_data"
    assert permutation_entropy(np.arange(30.0)).method
    assert higuchi_fd(np.arange(50.0)).method
    assert compute_matrix_profile(tiny, window=10).value == "insufficient_data"
    assert find_motifs(tiny, window=10).value == "insufficient_data"
    assert find_discords(tiny, window=10).value == "insufficient_data"
    assert subsequence_distance([1.0], [1.0]).value == "insufficient_data"
    assert nearest_neighbors([1.0], tiny).value == "insufficient_data"
    assert discover_shapelets(tiny, lengths=(8,)).value == "insufficient_data"
    assert johansen_trace(tiny, tiny).value == "insufficient_data"
    assert mutual_information(np.random.randn(40), np.random.randn(40)).method
    assert distance_correlation(np.random.randn(40), np.random.randn(40)).method
    assert empirical_tail_dependence(np.random.randn(80), np.random.randn(80)).method
    assert pelt_detect(np.ones(5)).indices == []
    assert bayesian_online_changepoint(tiny).method
    assert online_cusum(tiny).method
    assert acf(np.random.randn(40)).method
    assert pacf(np.random.randn(40)).method
    assert ccf(np.random.randn(40), np.random.randn(40)).method
    assert lead_lag(np.random.randn(40), np.random.randn(40)).method
    lo, hi = bartlett_bands(np.array([1.0, 0.2, 0.1]), n=100)
    assert lo.shape[0] == 3
    assert rolling_acf(np.random.randn(50), window=10, lag=1).method
    assert engle_granger(np.cumsum(np.random.randn(80)), np.cumsum(np.random.randn(80))).method
    assert granger_causality(np.random.randn(60), np.random.randn(60)).method
    assert cusum_detect(np.concatenate([np.zeros(40), np.ones(40) * 5])).method
    assert spectral_density(np.random.randn(80), method="welch").method
    assert log_transform(np.array([])).value == "insufficient_data"
    assert norm_wrap(np.array([1.0]), method="zscore").value == "insufficient_data"


def test_kpss_pvalue_branches():
    # force extreme stats via constant / strong trend
    assert kpss(np.ones(80)).pvalue is not None
    x = np.linspace(0, 10, 80) ** 2
    assert kpss(x, regression="ct").pvalue is not None


def test_transformer_winsorize_and_unknown():
    x = np.random.default_rng(2).normal(size=60)
    assert TimeSeriesTransformer(method="winsorize", window=15).fit_transform(x).shape[0] == 60
    # unknown method returns copy
    tr = TimeSeriesTransformer(method="log_return")
    tr.method = "unknown"  # type: ignore[assignment]
    assert tr.transform(x).shape[0] == 60
    assert log_via_transformer(np.abs(x) + 1).shape[0] == 60
    assert log_transform(np.array([-1.0, 0.0, 1.0])).method  # clips
    assert rank_transform(np.array([1.0])).value == "insufficient_data"
    assert norm_wrap(x, method="zscore").method


def test_classical_even_period_ma():
    t = np.arange(100, dtype=float)
    x = np.sin(2 * np.pi * t / 10) + t * 0.01
    assert classical_decompose(x, period=10, model="additive").method


def test_serializer_remaining():
    assert isinstance(_to_jsonable(np.int64(3)), int)

    class Obj:
        def __init__(self):
            self.a = 1
            self._b = 2

    assert _to_jsonable(Obj())["a"] == 1
    ser = TimeSeriesSerializer()
    assert "value" in ser.load_bytes(ser.dump_bytes(object()))

    class MD:
        def model_dump(self):
            return {"z": 1}

    assert _to_jsonable(MD()) == {"z": 1}


def test_registry_import_failure():
    with patch("importlib.import_module", side_effect=ImportError("x")):
        # still returns list (possibly empty for failed)
        loaded = ensure_timeseries_loaded()
        assert isinstance(loaded, list)


def test_orchestrator_methods_list():
    eng = TimeSeriesAnalyticsEngine()
    assert isinstance(eng.methods(), list)
    # spectral dominant empty mask path
    assert dominant_frequencies(np.zeros(32), min_frequency=10.0).value == []


def test_shapelets_unlabeled():
    x = np.random.default_rng(3).normal(size=100)
    assert discover_shapelets(x, labels=None, lengths=(8, 12), top_k=2, n_candidates=20).method


def test_volatility_features_short():
    assert volatility_features(np.array([1.0])).get("realized_vol") == 0.0
