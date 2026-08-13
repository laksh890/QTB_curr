"""Coverage gap fillers for Time-Series Analytics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from iqrp.app.timeseries.alignment.dtw import dtw_path
from iqrp.app.timeseries.alignment.shapelets import discover_shapelets
from iqrp.app.timeseries.alignment.soft_dtw import soft_dtw
from iqrp.app.timeseries.anomaly.matrix_profile import matrix_profile_anomalies
from iqrp.app.timeseries.anomaly.robust import mad_anomalies, robust_zscore_anomalies
from iqrp.app.timeseries.anomaly.statistical import zscore_anomalies
from iqrp.app.timeseries.autocorrelation.acf import rolling_acf
from iqrp.app.timeseries.autocorrelation.cross_correlation import lead_lag
from iqrp.app.timeseries.base import ChangePointResult, DecompositionResult, TemporalMode, finite_mask
from iqrp.app.timeseries.change_points.binary_segmentation import binseg_detect
from iqrp.app.timeseries.change_points.pelt import pelt_detect
from iqrp.app.timeseries.config import TimeSeriesSettings
from iqrp.app.timeseries.decomposition.classical import classical_decompose
from iqrp.app.timeseries.decomposition.seasonal import extract_seasonal
from iqrp.app.timeseries.decomposition.trend import extract_trend
from iqrp.app.timeseries.diagnostics import distribution_shift, heteroskedasticity, seasonality_diagnostics, structural_breaks
from iqrp.app.timeseries.features import (
    change_point_proximity,
    cycle_features,
    entropy_features,
    memory_features,
    spectral_features,
    trend_features,
    volatility_features,
)
from iqrp.app.timeseries.motifs.matrix_profile import compute_matrix_profile
from iqrp.app.timeseries.motifs.similarity import nearest_neighbors, subsequence_distance
from iqrp.app.timeseries.orchestrator import TimeSeriesAnalyticsEngine
from iqrp.app.timeseries.processes import from_market_simulator, simulate_process
from iqrp.app.timeseries.registry import get, register
from iqrp.app.timeseries.rolling import incremental_mean_var, temporal_contract
from iqrp.app.timeseries.serializer import TimeSeriesSerializer, _to_jsonable
from iqrp.app.timeseries.spectral.spectral_density import period_from_frequency, spectral_density
from iqrp.app.timeseries.stationarity import adf, kpss, phillips_perron, variance_ratio
from iqrp.app.timeseries.transforms import TimeSeriesTransformer
from iqrp.app.timeseries.transforms.differencing import differencing as difference, seasonal_differencing as seasonal_difference
from iqrp.app.timeseries.transforms.log_transform import log_transform
from iqrp.app.timeseries.transforms.normalization import robust_normalize, zscore_normalize, minmax_normalize
from iqrp.app.timeseries.transforms.rank_transform import rank_transform as rank
from iqrp.app.timeseries.transforms.returns import log_returns as lr2, simple_returns as sr2


def test_insufficient_data_branches():
    short = np.array([1.0, 2.0])
    assert adf(short).value == "insufficient_data"
    assert kpss(short).value == "insufficient_data"
    assert phillips_perron(short).value == "insufficient_data"
    assert variance_ratio(short).value == "insufficient_data"
    assert zscore_anomalies(short).value == "insufficient_data"
    assert robust_zscore_anomalies(short).value == "insufficient_data"
    assert pelt_detect(short).indices == []
    assert binseg_detect(short).indices == []


def test_transform_wrappers_and_training_only():
    x = np.random.default_rng(0).normal(size=80) + 10
    assert lr2(x).method
    assert sr2(x).method
    assert difference(x).method
    assert seasonal_difference(x, period=5).method
    assert log_transform(np.abs(x) + 1).method
    assert zscore_normalize(x, window=10).method
    assert robust_normalize(x, window=10).method
    assert rank(x, window=10).method
    tr = TimeSeriesTransformer(method="zscore", temporal_mode="training_only", window=20)
    tr.fit(x)
    assert tr.transform(x).shape[0] == x.size
    tr2 = TimeSeriesTransformer(method="robust", temporal_mode="training_only", window=20)
    tr2.fit(x)
    assert tr2.transform(x).shape[0] == x.size
    assert TimeSeriesTransformer(method="zscore", temporal_mode="expanding").fit_transform(x).shape[0] == x.size
    from iqrp.app.timeseries.transforms import normalize

    assert normalize(x, method="minmax", window=10).shape[0] == x.size
    assert normalize(x, method="robust", window=10).shape[0] == x.size
    assert minmax_normalize(x, window=10).method


def test_classical_multiplicative_and_trend_hp():
    t = np.arange(120, dtype=float)
    x = (10 + 0.1 * t) * (1 + 0.2 * np.sin(2 * np.pi * t / 12))
    c = classical_decompose(x, period=12, model="multiplicative")
    assert c.model == "multiplicative"
    assert extract_seasonal(x, period=12).method
    assert extract_trend(x, period=12, method="hp").method
    assert extract_trend(x, period=12, method="stl").method


def test_acf_rolling_and_lead_lag():
    x = np.random.default_rng(1).normal(size=100)
    y = np.roll(x, 3) + 0.1 * np.random.default_rng(2).normal(size=100)
    assert rolling_acf(x, window=30, lag=1).method
    assert lead_lag(x, y, max_lag=5).method


def test_spectral_density_helpers():
    x = np.sin(np.linspace(0, 20, 128))
    assert spectral_density(x).method
    assert period_from_frequency(0.1).method


def test_motif_similarity_and_mp_anomalies():
    x = np.random.default_rng(3).normal(size=120)
    mp = compute_matrix_profile(x, window=16)
    assert mp.method
    assert subsequence_distance(x[:16], x[20:36]).method
    assert nearest_neighbors(x[:16], x, top_k=2).method
    assert matrix_profile_anomalies(x, window=16).method
    assert mad_anomalies(x).method
    assert robust_zscore_anomalies(x, window=20).method
    assert zscore_anomalies(x, window=20).method


def test_shapelets_and_dtw_path():
    x = np.random.default_rng(4).normal(size=80)
    y = x + 0.5
    assert dtw_path(x, y).method
    assert soft_dtw(x, y).method
    labels = (np.arange(80) > 40).astype(int)
    assert discover_shapelets(x, labels=labels, lengths=(8, 10), top_k=2).method


def test_feature_facades():
    x = np.sin(np.linspace(0, 30, 150)) + 0.1 * np.random.default_rng(5).normal(size=150)
    assert "trend_strength" in trend_features(x)
    assert "seasonal_strength" in cycle_features(x)
    assert "realized_vol" in volatility_features(x)
    assert "shannon_entropy" in entropy_features(x)
    assert "hurst" in memory_features(x)
    assert "dominant_frequency" in spectral_features(x)
    assert "n_change_points" in change_point_proximity(x)


def test_diagnostics_facades():
    x = np.concatenate([np.zeros(60), np.ones(60) * 3])
    assert structural_breaks(x).method
    assert distribution_shift(x).method
    assert heteroskedasticity(np.random.default_rng(6).normal(size=80)).method
    assert seasonality_diagnostics(np.sin(np.linspace(0, 20, 120)), period=12).method


def test_serializer_and_registry():
    assert _to_jsonable(Path("/a")) == "/a"
    assert _to_jsonable(np.array([1.0])) == [1.0]
    assert _to_jsonable(np.float64(1.2)) == 1.2

    class TD:
        def to_dict(self):
            return {"a": 1}

    assert _to_jsonable(TD()) == {"a": 1}
    assert _to_jsonable(object())
    ser = TimeSeriesSerializer()
    eng = TimeSeriesAnalyticsEngine()
    assert ser.load_bytes(ser.dump_bytes(eng))
    assert ser.dump_bytes(AnalysisResult_like())

    @register("unit_test_fn")
    def _f(x):
        return x

    assert get("unit_test_fn")(1) == 1
    with pytest.raises(KeyError):
        get("___missing___")


class AnalysisResult_like:
    def to_dict(self):
        return {"ok": True}


def test_orchestrator_detect_branches_and_import():
    eng = TimeSeriesAnalyticsEngine()
    x = np.random.default_rng(7).normal(size=100)
    assert eng.detect(x, what="change_points")
    assert eng.detect(x, what="motifs")
    assert eng.detect(x, what="discords")
    with pytest.raises(ValueError):
        eng.detect(x, what="nope")
    with pytest.raises(ValueError):
        eng.correlate(x, kind="ccf")
    assert eng.decompose(x, method="classical").method
    assert eng.decompose(x, method="mstl").method
    assert eng.change_points(x, method="bayesian")
    assert eng.change_points(x, method="online")
    assert eng.anomalies(x, method="isolation_forest").method
    assert eng.anomalies(x, method="matrix_profile").method
    eng.import_state({"settings": {"seed": 1}, "fitted": True})
    eng.import_state({"settings": {"decomposition": {"method": "bad"}}})


def test_config_omegaconf_and_default_missing(tmp_path, monkeypatch):
    s = TimeSeriesSettings.from_mapping(OmegaConf.create({"seed": 3}))
    assert s.seed == 3
    monkeypatch.setattr(
        "iqrp.app.timeseries.config._default_config_path",
        lambda: tmp_path / "missing.yaml",
    )
    assert TimeSeriesSettings.default().seed == 42


def test_kpss_ct_and_finite_mask():
    x = np.linspace(0, 1, 100) + np.random.default_rng(8).normal(0, 0.01, 100)
    assert kpss(x, regression="ct").method == "kpss"
    assert finite_mask(np.array([1.0, np.nan])).sum() == 1


def test_processes_fallback_and_unknown():
    with patch(
        "iqrp.app.simulation.base.simulator.MarketSimulator",
        side_effect=RuntimeError("x"),
    ):
        out = from_market_simulator(50)
        assert "series" in out
    assert "series" in simulate_process("unknown_kind", 20, seed=1)  # type: ignore[arg-type]


def test_result_to_dict_arrays():
    dec = DecompositionResult(
        method="t",
        trend=np.ones(3),
        seasonal=np.zeros(3),
        residual=np.zeros(3),
        observed=np.ones(3),
    )
    assert dec.to_dict()["trend"] == [1.0, 1.0, 1.0]
    cp = ChangePointResult(method="c", indices=[1], scores=np.array([0.5]))
    assert cp.to_dict()["scores"] == [0.5]


def test_rolling_temporal_and_welford():
    assert temporal_contract("rolling") == TemporalMode.ROLLING
    assert temporal_contract("nope") == TemporalMode.FULL_SAMPLE
    n, m, m2 = incremental_mean_var(0, 0.0, 0.0, 1.0)
    assert n == 1


def test_mstl_skip_large_period():
    from iqrp.app.timeseries.decomposition.mstl import mstl_decompose

    x = np.random.default_rng(9).normal(size=40)
    assert mstl_decompose(x, periods=(24, 200)).method == "mstl"
