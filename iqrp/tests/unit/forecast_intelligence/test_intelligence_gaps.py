"""Coverage gap fillers for Forecast Intelligence."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.forecasting.intelligence.benchmark import BenchmarkResult, _aggregate_folds, make_splits
from iqrp.app.forecasting.intelligence.calibration import Calibrator, apply_calibration, fit_calibrator
from iqrp.app.forecasting.intelligence.config import BenchmarkConfig, EnsembleConfig, IntelligenceSettings
from iqrp.app.forecasting.intelligence.drift import DriftReport, detect_drift
from iqrp.app.forecasting.intelligence.ensemble import bayesian_model_averaging, dynamic_ensemble_selection, voting_ensemble
from iqrp.app.forecasting.intelligence.gating import moe_combine
from iqrp.app.forecasting.intelligence.orchestrator import ForecastIntelligenceEngine
from iqrp.app.forecasting.intelligence.processes import simulate_market_frame, feature_names
from iqrp.app.forecasting.intelligence.ranking import RankedModel, composite_score
from iqrp.app.forecasting.intelligence.registry import DiscoveredModel, create_model
from iqrp.app.forecasting.intelligence.retraining import restore_checkpoint, decide_retrain
from iqrp.app.forecasting.intelligence.config import RetrainConfig, RankingConfig
from iqrp.app.forecasting.intelligence.selector import select_best
from iqrp.app.forecasting.intelligence.stacking import stack_predictions
from iqrp.app.forecasting.intelligence.uncertainty import ensemble_uncertainty, model_agreement


FEATS = feature_names(3)


def test_benchmark_result_dict():
    br = BenchmarkResult("m", "baseline", {"rmse": 1.0})
    assert br.to_dict()["name"] == "m"
    assert _aggregate_folds([{"a": 1.0}, {"a": 3.0}])["a"] == 2.0
    assert _aggregate_folds([]) == {}


def test_discovered_model_dict():
    d = DiscoveredModel("mock", "baseline", "1", True, True, True)
    assert d.to_dict()["name"] == "mock"


def test_calibrator_dirichlet_matrix():
    cal = Calibrator("dirichlet", {"temperature": 1.5})
    p = np.array([[0.2, 0.8], [0.4, 0.6]])
    out = cal.transform(p)
    assert out.shape == (2, 2)
    assert apply_calibration(None, np.array([1.0])).tolist() == [1.0]


def test_platt_on_probabilities():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    cal = fit_calibrator(y, s, method="platt")
    assert cal is not None


def test_empty_ensemble_uncertainty():
    assert ensemble_uncertainty({})["mean"].size == 0
    assert model_agreement({"a": np.ones(3)}) == 1.0


def test_moe_per_row_weights():
    preds = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    w = np.array([[0.5, 0.5], [0.2, 0.8]])
    out = moe_combine(preds, gate_weights=w)
    assert out.shape == (2,)


def test_stack_with_meta_weights():
    preds = {"a": np.array([1.0, 2.0]), "b": np.array([2.0, 3.0])}
    assert stack_predictions(preds, meta_weights=np.array([0.5, 0.5])).size == 2
    assert stack_predictions({}).size == 0


def test_voting_and_dynamic():
    preds = {"a": np.array([1.0, -1.0]), "b": np.array([1.0, 1.0])}
    assert voting_ensemble(preds).size == 2
    assert dynamic_ensemble_selection(preds, {"a": 0.1, "b": 0.9}, top_k=1).size == 2
    assert bayesian_model_averaging(preds, {"a": 0.1, "b": 0.5}).size == 2


def test_drift_report_disabled():
    r = detect_drift(
        ref_features=np.ones((20, 2)),
        cur_features=np.ones((20, 2)) * 10,
        config=IntelligenceSettings.from_mapping({"drift": {"enabled": False}}).drift,
    )
    assert r.triggered is False
    assert DriftReport({}, 0, 0, 0, 0, False).to_dict()["triggered"] is False


def test_decide_retrain_performance_drift():
    from iqrp.app.forecasting.intelligence.drift import DriftReport

    drift = DriftReport({"f0": 1.0}, 1.0, 1.0, 1.0, 1.0, True, ["feature_drift"])
    d = decide_retrain(n_updates=1, config=RetrainConfig(mode="performance"), drift=drift)
    assert d.should_retrain
    d2 = decide_retrain(n_updates=1, config=RetrainConfig(mode="performance"), performance_degraded=True)
    assert d2.should_retrain


def test_restore_checkpoint_noop():
    m = create_model("mock")
    assert restore_checkpoint(m, {}) is m


def test_select_best_with_regime():
    frame = simulate_market_frame(120, kind="regime_switching", n_features=3, rng=np.random.default_rng(4))
    settings = IntelligenceSettings.from_mapping(
        {
            "benchmark": {"method": "walk_forward", "n_splits": 2, "train_size": 40, "test_size": 15, "parallel": False},
            "ensemble": {"method": "none"},
        }
    )
    sel = select_best(frame, feature_columns=FEATS, target_column="target", settings=settings, candidates=["mock"])
    assert sel.best_model == "mock"
    assert sel.to_dict()["best_horizon"] >= 1


def test_engine_import_export_roundtrip(tmp_path):
    frame = simulate_market_frame(100, n_features=3, rng=np.random.default_rng(5))
    eng = ForecastIntelligenceEngine(
        IntelligenceSettings.from_mapping(
            {"benchmark": {"parallel": False, "n_splits": 2, "train_size": 40, "test_size": 15}, "ensemble": {"method": "none"}}
        )
    )
    eng.fit(frame, feature_columns=FEATS, candidates=["mock"], run_selection=False)
    eng.calibrate(frame, method="temperature")
    state = eng.export_state()
    eng2 = ForecastIntelligenceEngine()
    eng2.import_state(state)
    assert eng2.best_model() == "mock"


def test_composite_empty_weights():
    cfg = RankingConfig(weights={})
    assert np.isfinite(composite_score({"rmse": 1.0}, cfg))


def test_make_splits_fallback_small():
    splits = make_splits(30, BenchmarkConfig(method="walk_forward", train_size=100, test_size=50))
    assert isinstance(splits, list)


def test_build_ensemble_none():
    from iqrp.app.forecasting.intelligence.ensemble import build_ensemble

    preds = {"a": np.array([1.0, 2.0])}
    out = build_ensemble(preds, config=EnsembleConfig(method="none"))
    assert out.tolist() == [1.0, 2.0]
