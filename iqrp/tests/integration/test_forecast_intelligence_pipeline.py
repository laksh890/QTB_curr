"""Integration / stress tests for Forecast Intelligence."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.forecasting.intelligence import ForecastIntelligenceEngine, IntelligenceSettings
from iqrp.app.forecasting.intelligence.processes import feature_names, simulate_market_frame

FEATS = feature_names(4)


def _eng(**kw) -> ForecastIntelligenceEngine:
    base = {
        "benchmark": {
            "method": "walk_forward",
            "n_splits": 2,
            "train_size": 60,
            "test_size": 20,
            "parallel": False,
        },
        "ensemble": {"method": "weighted", "top_k": 1},
        "automl": {"method": "none"},
        "retrain": {"mode": "drift", "window": 100, "warm_start": True},
        "drift": {
            "enabled": True,
            "feature_psi_threshold": 0.01,
            "prediction_ks_threshold": 0.01,
            "performance_drop": 0.05,
        },
    }
    base.update(kw)
    return ForecastIntelligenceEngine(IntelligenceSettings.from_mapping(base))


@pytest.mark.parametrize(
    "kind",
    ["trending", "mean_reverting", "volatile", "regime_switching", "cross_asset"],
)
def test_simulation_validation_selection(kind):
    frame = simulate_market_frame(160, kind=kind, n_features=4, rng=np.random.default_rng(10))
    eng = _eng()
    eng.fit(frame, feature_columns=FEATS, candidates=["mock"])
    assert eng.best_model() == "mock"
    fc = eng.forecast(frame, horizon=5)
    assert fc.values.size == 5
    assert len(eng.leaderboard()) >= 1


def test_ensemble_superiority_path():
    frame = simulate_market_frame(150, kind="volatile", n_features=4, rng=np.random.default_rng(11))
    eng = _eng(ensemble={"method": "median", "top_k": 1})
    eng.fit(frame, feature_columns=FEATS, candidates=["mock"])
    pred = eng.ensemble(frame, method="median")
    assert pred.size == frame.height
    fc = eng.forecast(frame, horizon=4)
    assert "members" in fc.metadata or fc.model_name in {"mock", "ensemble"}


def test_drift_triggers_retrain():
    frame = simulate_market_frame(
        140, kind="regime_switching", n_features=4, rng=np.random.default_rng(12)
    )
    eng = _eng()
    eng.fit(frame, feature_columns=FEATS, candidates=["mock"])
    # shift features strongly
    shifted = frame.with_columns([frame[c] * 5 + 10 for c in FEATS])
    report = eng.detect_drift(shifted)
    assert report.triggered or report.covariate_shift > 0
    decision = eng.retrain(shifted)
    assert decision.to_dict()["mode"] == "drift"


def test_automl_fit_path():
    frame = simulate_market_frame(120, kind="trending", n_features=4, rng=np.random.default_rng(13))
    eng = _eng(automl={"method": "grid", "n_trials": 3})
    eng.fit(frame, feature_columns=FEATS, candidates=["mock"], run_automl=True)
    assert eng._fitted
    assert eng.predict_proba(frame).shape[0] == frame.height


def test_benchmark_methods_integration():
    frame = simulate_market_frame(
        130, kind="mean_reverting", n_features=4, rng=np.random.default_rng(14)
    )
    for method in ("rolling", "purged_kfold", "embargo"):
        eng = _eng(
            benchmark={
                "method": method,
                "n_splits": 2,
                "train_size": 50,
                "test_size": 15,
                "parallel": False,
            }
        )
        rows = eng.benchmark(frame, feature_columns=FEATS, candidates=["mock"])
        assert rows[0]["name"] == "mock"


def test_online_monitor_stress():
    frame = simulate_market_frame(100, kind="volatile", n_features=4, rng=np.random.default_rng(15))
    eng = _eng()
    eng.fit(frame, feature_columns=FEATS, candidates=["mock"], run_selection=False)
    pred = eng.predict(frame)
    for i in range(30):
        eng.monitor(y_true=float(frame["target"][i]), y_pred=float(pred[i]))
    snap = eng.monitor()
    assert snap.n_observations >= 30
