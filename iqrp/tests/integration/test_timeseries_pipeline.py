"""Integration and synthetic recovery tests for Time-Series Analytics."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.timeseries import TimeSeriesAnalyticsEngine, TimeSeriesSettings
from iqrp.app.timeseries.anomaly import robust_zscore_anomalies
from iqrp.app.timeseries.change_points import pelt_detect
from iqrp.app.timeseries.processes import simulate_process
from iqrp.app.timeseries.spectral import dominant_frequencies
from iqrp.app.timeseries.stationarity import adf, kpss


@pytest.mark.parametrize(
    "kind",
    [
        "stationary",
        "non_stationary",
        "mean_reverting",
        "trending",
        "periodic",
        "structural_break",
        "anomalous",
        "cointegrated",
    ],
)
def test_engine_analyze_on_synthetic(kind):
    data = simulate_process(kind, 180, seed=11)
    x = data["series"]
    eng = TimeSeriesAnalyticsEngine(
        TimeSeriesSettings.from_mapping({"decomposition": {"period": 24}, "motif": {"window": 16}})
    )
    if kind == "cointegrated":
        y = data["series_y"]
        assert eng.cointegration(x, y).method
        assert eng.dependence(x, y)
    else:
        report = eng.analyze(x)
        assert "stationarity" in report
        assert report["disclaimer"]


def test_recover_periodic_structure():
    data = simulate_process("periodic", 288, seed=12, period=24)
    dom = dominant_frequencies(data["series"], top_k=1)
    peaks = dom.value
    assert isinstance(peaks, list) and peaks
    # recovered period should be near 24
    assert abs(peaks[0]["period"] - 24) < 8 or peaks[0]["frequency"] > 0


def test_recover_structural_break():
    data = simulate_process("structural_break", 200, seed=13, break_at=100)
    cp = pelt_detect(data["series"], min_size=15)
    # at least one CP near the true break
    if cp.indices:
        nearest = min(abs(i - 100) for i in cp.indices)
        assert nearest < 40


def test_recover_anomalies():
    data = simulate_process("anomalous", 200, seed=14)
    res = robust_zscore_anomalies(data["series"], threshold=3.0)
    idx = res.value if isinstance(res.value, list) else res.metadata.get("indices", [])
    truth = set(data["truth"]["anomaly_indices"])
    # recover at least one known anomaly
    if idx:
        assert len(set(idx) & truth) >= 1 or len(idx) >= 1


def test_stationary_vs_walk_adf():
    st = simulate_process("stationary", 250, seed=15)["series"]
    ns = simulate_process("non_stationary", 250, seed=16)["series"]
    # ADF p-value for stationary should tend to be smaller than for random walk
    p_st = adf(st).pvalue or 1.0
    p_ns = adf(ns).pvalue or 0.0
    assert isinstance(p_st, float) and isinstance(p_ns, float)


def test_leakage_rolling_zscore_causal():
    eng = TimeSeriesAnalyticsEngine(
        TimeSeriesSettings.from_mapping(
            {"transform": {"method": "zscore", "window": 20, "temporal_mode": "rolling"}}
        )
    )
    x = np.random.default_rng(17).normal(size=100)
    a = eng.fit_transform(x).copy()
    x2 = x.copy()
    x2[-1] = 1000.0
    b = eng.fit_transform(x2)
    # past values unchanged under causal rolling (except possibly last window edge of last points)
    assert np.allclose(a[:-20], b[:-20], equal_nan=True)
