"""Rolling retraining: triggers, registry, snapshots, retrainer."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.backtesting.rolling_retraining import (
    CompositeTrigger,
    DriftTrigger,
    FeatureSnapshotStore,
    ModelRegistry,
    ParameterSnapshotStore,
    PerformanceTrigger,
    RegimeTrigger,
    RetrainSchedule,
    RollingRetrainer,
    TimeTrigger,
)


def test_time_performance_drift_regime_triggers() -> None:
    tt = TimeTrigger(every=10)
    d = tt.evaluate({"t": 20, "last_retrain_t": 5})
    assert d.should_retrain
    assert d.to_dict()["should_retrain"]

    idle = tt.evaluate({"t": 8, "last_retrain_t": 5})
    assert not idle.should_retrain

    pt = PerformanceTrigger(metric="sharpe", min_value=1.0)
    bad = pt.evaluate({"sharpe": 0.1})
    assert bad.should_retrain
    ok = pt.evaluate({"sharpe": 2.0})
    assert not ok.should_retrain
    from_series = pt.evaluate({"performance": np.array([0.0, -0.01, -0.02] * 10)})
    assert from_series.kind == "performance"
    missing = pt.evaluate({})
    assert not missing.should_retrain

    dt = DriftTrigger(threshold=0.1)
    fire = dt.evaluate({"drift_score": 0.5})
    assert fire.should_retrain
    computed = dt.evaluate({"feature_ref": np.zeros(20), "feature_cur": np.ones(20)})
    assert computed.should_retrain or computed.details
    no = dt.evaluate({})
    assert not no.should_retrain

    rt = RegimeTrigger()
    change = rt.evaluate({"regime": "high", "previous_regime": "low"})
    assert change.should_retrain
    same = rt.evaluate({"regime": "low", "previous_regime": "low"})
    assert not same.should_retrain
    miss = rt.evaluate({})
    assert not miss.should_retrain


def test_composite_and_schedule() -> None:
    comp = CompositeTrigger(
        triggers=[TimeTrigger(every=5), DriftTrigger(threshold=0.01)], combine="any"
    )
    d = comp.evaluate({"t": 10, "last_retrain_t": 0, "drift_score": 0.0})
    assert d.should_retrain
    empty = CompositeTrigger(triggers=[], combine="all")
    assert not empty.evaluate({}).should_retrain

    sched = RetrainSchedule(every=10, min_bars_between=3, max_retrains=2)
    assert sched.should_retrain({"t": 20, "last_retrain_t": 5}).should_retrain
    blocked = RetrainSchedule(every=1, min_bars_between=10)
    assert not blocked.should_retrain({"t": 5, "last_retrain_t": 0}).should_retrain
    limited = RetrainSchedule(every=1, max_retrains=1)
    limited.record_retrain()
    assert not limited.should_retrain({"t": 100, "last_retrain_t": 0}).should_retrain
    assert limited.retrain_count == 1

    multi = RetrainSchedule(triggers=[TimeTrigger(every=5)], combine="all")
    assert multi.trigger is not None


def test_model_registry_and_snapshots() -> None:
    reg = ModelRegistry()
    s1 = reg.register({"w": 1}, trained_through=10, trigger="time")
    s2 = reg.register({"w": 2}, trained_through=20, activate=False)
    assert reg.size == 2
    assert reg.active_version == 1
    reg.activate(2)
    assert reg.active().version == 2
    assert reg.get(1).trained_through == 10
    assert reg.latest().version == 2
    assert reg.versions() == [1, 2]
    assert len(reg.history()) == 2
    assert s1.to_dict()["version"] == 1
    reg.clear()
    assert reg.size == 0
    with pytest.raises(KeyError):
        ModelRegistry().activate(9)

    feats = FeatureSnapshotStore()
    fs = feats.save(np.ones((5, 2)), start=0, end=5, columns=["a", "b"])
    assert fs.version == 1
    assert feats.get(fs.version) is not None

    params = ParameterSnapshotStore()
    ps = params.save({"lr": 0.01}, source="time")
    assert ps.version == 1
    assert params.get(ps.version).params["lr"] == 0.01


def test_rolling_retrainer_run() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = rng.normal(size=80)

    def train_fn(X_tr, y_tr, params):
        return {"mu": float(np.mean(y_tr)) if y_tr is not None else 0.0}

    def predict_fn(model, X_te):
        return np.full(len(X_te), model["mu"])

    def score_fn(model, X_te, y_te):
        pred = predict_fn(model, X_te)
        return {"mse": float(np.mean((pred - y_te) ** 2)), "n": float(len(y_te))}

    rr = RollingRetrainer(schedule=RetrainSchedule(every=15), train_window=30, origin=0)
    report = rr.run(X=X, y=y, train_fn=train_fn, predict_fn=predict_fn, score_fn=score_fn)
    assert report["n_models"] >= 1
    assert report["events"]
    assert rr.active_model() is not None

    # predict_at causal
    snap = rr.registry.active()
    t = int(snap.trained_through) + 1
    pred = rr.predict_at(t, X=X, predict_fn=predict_fn)
    assert pred is not None

    with pytest.raises(ValueError):
        RollingRetrainer(train_window=0)

    start, end = rr.training_slice(40)
    assert end == 40 and start < end


def test_rolling_retrainer_no_future_training() -> None:
    X = np.arange(30).reshape(30, 1).astype(float)
    y = np.arange(30).astype(float)
    rr = RollingRetrainer(schedule=RetrainSchedule(every=100), origin=0)

    def train_fn(X_tr, y_tr, params):
        return {"n": len(X_tr)}

    snap = rr.maybe_retrain(10, X=X, y=y, train_fn=train_fn, force=True)
    assert snap is not None
    assert snap.trained_through < 10

    with pytest.raises(ValueError):
        rr.predict_at(snap.trained_through, X=X, predict_fn=lambda m, x: x)
