"""Core unit tests for Institutional Forecast Intelligence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.core.exceptions import ModelError
from iqrp.app.forecasting.intelligence import (
    ForecastIntelligenceEngine,
    IntelligenceSettings,
    discover_engine_modules,
    list_discovered_models,
    load_discovered_engines,
)
from iqrp.app.forecasting.intelligence.automl import optimize_model
from iqrp.app.forecasting.intelligence.benchmark import (
    benchmark_candidates,
    benchmark_model,
    make_splits,
)
from iqrp.app.forecasting.intelligence.blending import blend_predictions, holdout_blend_weights
from iqrp.app.forecasting.intelligence.calibration import apply_calibration, fit_calibrator
from iqrp.app.forecasting.intelligence.config import (
    BenchmarkConfig,
    IntelligenceSettings as Settings,
    RankingConfig,
    RetrainConfig,
)
from iqrp.app.forecasting.intelligence.deployment import DeploymentManager
from iqrp.app.forecasting.intelligence.diagnostics import diagnose_leaderboard, diagnose_residuals
from iqrp.app.forecasting.intelligence.drift import (
    detect_drift,
    ks_statistic,
    population_stability_index,
)
from iqrp.app.forecasting.intelligence.ensemble import (
    build_ensemble,
    median_ensemble,
    weighted_average,
)
from iqrp.app.forecasting.intelligence.gating import moe_combine, regime_gate_logits, softmax
from iqrp.app.forecasting.intelligence.monitoring import ForecastMonitor
from iqrp.app.forecasting.intelligence.optimization import run_optimization
from iqrp.app.forecasting.intelligence.processes import feature_names, simulate_market_frame
from iqrp.app.forecasting.intelligence.ranking import composite_score, compute_metrics, rank_models
from iqrp.app.forecasting.intelligence.registry import create_model
from iqrp.app.forecasting.intelligence.retraining import (
    checkpoint_model,
    decide_retrain,
    retrain_model,
)
from iqrp.app.forecasting.intelligence.routing import build_routing_table, route_model
from iqrp.app.forecasting.intelligence.serializer import IntelligenceSerializer
from iqrp.app.forecasting.intelligence.stacking import fit_stacker, stack_predictions
from iqrp.app.forecasting.intelligence.tuning import TuningHistory, TuningTrial, build_search_space
from iqrp.app.forecasting.intelligence.uncertainty import (
    ensemble_uncertainty,
    forecast_distribution,
    model_agreement,
    prediction_intervals,
)
from iqrp.app.forecasting.intelligence.visualization import (
    drift_chart,
    forecast_chart,
    leaderboard_chart,
    residual_hist,
)

FEATS = feature_names(3)


def _settings(**overrides) -> IntelligenceSettings:
    base = {
        "benchmark": {
            "method": "walk_forward",
            "n_splits": 2,
            "train_size": 50,
            "test_size": 15,
            "parallel": False,
        },
        "ensemble": {"method": "weighted", "top_k": 1},
        "automl": {"method": "none"},
        "calibration": {"enabled": False, "method": "none"},
        "retrain": {"mode": "performance", "window": 80, "warm_start": True},
        "drift": {
            "enabled": True,
            "feature_psi_threshold": 0.05,
            "prediction_ks_threshold": 0.05,
            "performance_drop": 0.1,
        },
        "monitoring": {"enabled": True, "window": 50},
    }
    base.update(overrides)
    return IntelligenceSettings.from_mapping(base)


@pytest.fixture
def frame():
    return simulate_market_frame(120, kind="trending", n_features=3, rng=np.random.default_rng(1))


@pytest.fixture
def engine(frame):
    eng = ForecastIntelligenceEngine(_settings())
    eng.fit(frame, feature_columns=FEATS, candidates=["mock"], run_selection=True)
    return eng


def test_discovery_loads_modules():
    mods = discover_engine_modules()
    assert any("mock" in m or "registry" in m for m in mods)
    loaded = load_discovered_engines(mods[:5] if len(mods) > 5 else mods)
    assert isinstance(loaded, list)
    models = list_discovered_models(_settings(discovery={"max_candidates": 3}))
    assert len(models) <= 3
    assert any(m.name == "mock" for m in list_discovered_models())


def test_settings_hydra_and_invalid():
    s = IntelligenceSettings.default()
    assert s.seed == 42
    s2 = IntelligenceSettings.from_hydra(overrides=["seed=7"])
    assert s2.seed == 7
    with pytest.raises(Exception):
        IntelligenceSettings.from_mapping({"benchmark": {"method": "not_a_method"}})


def test_splits_all_methods():
    for method in (
        "walk_forward",
        "rolling",
        "time_series_split",
        "nested_cv",
        "purged_kfold",
        "embargo",
    ):
        splits = make_splits(
            100, BenchmarkConfig(method=method, n_splits=3, train_size=40, test_size=10)
        )
        assert isinstance(splits, list)


def test_ranking_and_metrics(frame):
    y = frame["target"].to_numpy()
    p = y + 0.01
    m = compute_metrics(
        y, p, probabilities=np.column_stack([1 - (p > 0), (p > 0).astype(float)]), latency_ms=1.0
    )
    assert "rmse" in m and "brier" in m
    score = composite_score(m, RankingConfig())
    assert np.isfinite(score)
    ranked = rank_models(
        [{"name": "a", "metrics": m}, {"name": "b", "metrics": {**m, "rmse": m["rmse"] + 1}}]
    )
    assert ranked[0].name == "a"
    assert ranked[0].to_dict()["rank"] == 1


def test_benchmark_mock(frame):
    s = _settings()
    res = benchmark_model("mock", frame, feature_columns=FEATS, target_column="target", settings=s)
    assert res.name == "mock"
    assert "rmse" in res.metrics
    results = benchmark_candidates(
        frame, feature_columns=FEATS, target_column="target", settings=s, candidates=["mock"]
    )
    assert len(results) == 1


def test_engine_api(engine, frame, tmp_path: Path):
    pred = engine.predict(frame)
    assert pred.shape[0] == frame.height
    proba = engine.predict_proba(frame)
    assert proba.ndim == 2
    fc = engine.forecast(frame, horizon=3)
    assert fc.values.size == 3
    intervals = engine.forecast_interval(frame, horizon=3)
    assert len(intervals) == 3
    assert engine.best_model() == "mock"
    assert len(engine.leaderboard()) >= 1
    assert engine.leaderboard(by="asset", frame=frame)
    assert engine.leaderboard(by="regime", frame=frame)
    assert engine.leaderboard(by="timeframe", frame=frame)
    assert engine.leaderboard(by="feature_set", frame=frame)
    ens = engine.ensemble(frame)
    assert ens.size == frame.height
    cal = engine.calibrate(frame, method="platt")
    assert cal is not None
    snap = engine.monitor(y_true=0.0, y_pred=float(pred[-1]))
    assert snap.n_observations >= 1
    drift = engine.detect_drift(frame)
    assert isinstance(drift.triggered, bool)
    diag = engine.diagnose(frame)
    assert "residuals" in diag
    viz = engine.visualize(frame)
    assert "leaderboard" in viz
    dist = engine.distribution(frame, horizon=2, n_samples=10)
    assert dist.shape == (10, 2)
    path = engine.save(tmp_path / "fi.json")
    loaded = ForecastIntelligenceEngine.load(path)
    assert loaded.best_model() == "mock"
    dep = engine.deploy(name="test")
    assert dep["status"] == "active"
    assert engine.discovered_models()


def test_engine_not_fitted():
    eng = ForecastIntelligenceEngine(_settings())
    with pytest.raises(ModelError):
        eng.predict(simulate_market_frame(40, n_features=3))


def test_retrain_and_checkpoint(engine, frame):
    model = create_model("mock")
    model.fit(frame, feature_columns=FEATS, target_column="target")
    ckpt = checkpoint_model(model)
    assert isinstance(ckpt, dict)
    decision = engine.retrain(frame, force=True)
    assert decision.should_retrain
    d2 = decide_retrain(n_updates=100, config=RetrainConfig(mode="scheduled", schedule_every=100))
    assert d2.should_retrain
    d3 = decide_retrain(n_updates=1, config=RetrainConfig(mode="none"))
    assert not d3.should_retrain
    d4 = decide_retrain(n_updates=1, config=RetrainConfig(mode="rolling"))
    assert d4.should_retrain
    retrain_model(
        model,
        frame,
        feature_columns=FEATS,
        target_column="target",
        config=RetrainConfig(warm_start=True),
    )


def test_automl_methods(frame):
    s = _settings(automl={"method": "random", "n_trials": 3})
    params = optimize_model(
        "mock", frame, feature_columns=FEATS, target_column="target", settings=s
    )
    assert isinstance(params, dict)
    for method in ("grid", "hyperband", "successive_halving", "pbt", "bayesian"):
        s2 = _settings(automl={"method": method, "n_trials": 3})
        out = run_optimization(
            "mock", frame, feature_columns=FEATS, target_column="target", settings=s2
        )
        assert isinstance(out, dict)
    hist = TuningHistory()
    hist.add(TuningTrial(params={"drift": 0.0}, score=1.0))
    assert hist.best().score == 1.0
    assert build_search_space(family="tree")
    assert build_search_space(family="neural")
    assert build_search_space(family="transformer")
    assert build_search_space(family="volatility")
    assert build_search_space(family="statistical")


def test_ensemble_stack_blend_gate():
    preds = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([1.5, 2.5, 3.5])}
    assert weighted_average(preds).shape == (3,)
    assert median_ensemble(preds).shape == (3,)
    from iqrp.app.forecasting.intelligence.config import EnsembleConfig

    for method in ("weighted", "median", "bma", "voting", "stacking", "blending", "moe", "dynamic"):
        out = build_ensemble(
            preds, config=EnsembleConfig(method=method), scores={"a": 1.0, "b": 2.0}
        )
        assert out.size == 3
    assert stack_predictions(preds, meta_features=np.array([1.0, 2.0, 3.0])).size == 3
    assert (
        fit_stacker(np.column_stack([preds["a"], preds["b"]]), np.array([1.0, 2.0, 3.0])).size == 2
    )
    assert blend_predictions(preds, scores={"a": 1.0, "b": 2.0}).size == 3
    assert holdout_blend_weights(preds, np.array([1.0, 2.0, 3.0]))
    assert moe_combine(preds, gate_logits=np.array([1.0, 0.0])).size == 3
    assert moe_combine(preds, gate_weights=np.array([0.7, 0.3])).size == 3
    assert softmax(np.array([1.0, 2.0])).sum() == pytest.approx(1.0)
    assert regime_gate_logits(np.array([0, 1, 0]), 2).shape == (3, 2)


def test_calibration_methods():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=float)
    s = np.array([-1, 2, -0.5, 1.5, 0.8, -0.2, 1.1, -0.9])
    for method in ("temperature", "platt", "isotonic", "dirichlet"):
        cal = fit_calibrator(y, s, method=method)
        assert cal is not None
        out = apply_calibration(cal, s)
        assert out.shape[0] == s.shape[0]
    assert fit_calibrator(y, s, method="none") is None


def test_uncertainty_and_viz():
    preds = {"a": np.ones(5), "b": np.ones(5) * 1.1}
    unc = ensemble_uncertainty(preds)
    assert "mean" in unc
    assert model_agreement(preds) > 0
    intervals = prediction_intervals(np.zeros(3), residual_std=0.1)
    assert len(intervals) == 3
    dist = forecast_distribution(np.zeros(2), 0.1, n_samples=5)
    assert dist.shape == (5, 2)
    from iqrp.app.forecasting.intelligence.ranking import RankedModel

    chart = leaderboard_chart([RankedModel("m", {"rmse": 1}, 1.0, 1)])
    assert chart["type"] == "bar"
    assert forecast_chart([1, 2], np.array([0.0, 1.0]), np.array([0.1, 0.9]))
    assert drift_chart({"f0": 0.1})
    assert residual_hist(np.random.default_rng(0).normal(size=50))


def test_drift_psi_ks():
    a = np.random.default_rng(0).normal(size=100)
    b = a + 2.0
    assert population_stability_index(a, b) > 0
    assert ks_statistic(a, b) > 0
    report = detect_drift(
        ref_features=np.column_stack([a, a]),
        cur_features=np.column_stack([b, b]),
        ref_preds=a,
        cur_preds=b,
        ref_target=a,
        cur_target=b,
        ref_metric=0.1,
        cur_metric=0.5,
    )
    assert report.triggered
    assert report.to_dict()["triggered"]


def test_routing():
    frame = simulate_market_frame(
        60, kind="regime_switching", n_features=3, rng=np.random.default_rng(2)
    )
    table = build_routing_table("mock", regime_models={"0": "mock"}, high_vol_model="mock")
    name = route_model(frame, table, config=_settings().routing, confidence=0.1)
    assert name == "mock"


def test_monitor_and_deployment(tmp_path: Path, engine):
    mon = ForecastMonitor(_settings().monitoring)
    mon.record(y_true=0.1, y_pred=0.2, latency_ms=5.0, features=np.ones(3))
    mon.record(y_true=0.0, y_pred=0.1, latency_ms=8.0, features=np.ones(3) * 2)
    snap = mon.snapshot()
    assert snap.to_dict()["n_observations"] == 2
    mgr = DeploymentManager(tmp_path / "deploy")
    rec = mgr.deploy(engine, name="e1")
    assert mgr.active is not None
    mgr.deploy(engine, name="e2")
    assert mgr.rollback() is not None
    ser = IntelligenceSerializer()
    assert ser.load_bytes(ser.dump_bytes(engine))["best_model"] == "mock"


def test_diagnostics_residuals():
    y = np.linspace(0, 1, 50)
    p = y + 0.01
    rep = diagnose_residuals(y, p, lower=p - 0.1, upper=p + 0.1)
    assert rep.coverage_95 is not None
    from iqrp.app.forecasting.intelligence.ranking import RankedModel

    assert diagnose_leaderboard([RankedModel("a", {"rmse": 1}, 1.0)])["top"] == "a"
    assert diagnose_leaderboard([])["n_models"] == 0


def test_market_kinds():
    for kind in ("trending", "mean_reverting", "volatile", "regime_switching", "cross_asset"):
        fr = simulate_market_frame(80, kind=kind, n_features=3, rng=np.random.default_rng(3))
        assert fr.height == 80
        assert "target" in fr.columns


def test_parallel_benchmark(frame):
    s = _settings(
        benchmark={"method": "time_series_split", "n_splits": 2, "parallel": True, "max_workers": 2}
    )
    results = benchmark_candidates(
        frame,
        feature_columns=FEATS,
        target_column="target",
        settings=s,
        candidates=["mock", "mock"],
    )
    assert len(results) == 2
