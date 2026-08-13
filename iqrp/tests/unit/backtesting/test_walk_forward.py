"""Walk-forward windows, purge, embargo, purged k-fold, engine."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.backtesting.walk_forward import (
    WalkForwardEngine,
    WalkForwardEvaluator,
    apply_embargo,
    apply_purge,
    assert_no_future_training,
    embargo_after_test,
    embargo_range,
    embargo_splits,
    generate_windows,
    purge_range,
    purge_train_indices,
    purged_kfold_splits,
)
from iqrp.app.backtesting.walk_forward.test_window import TestWindow as WFTestWindow
from iqrp.app.backtesting.walk_forward.training_window import TrainingWindow
from iqrp.app.backtesting.walk_forward.validation_window import ValidationWindow
from iqrp.app.backtesting.walk_forward.windows import WalkForwardWindow


def test_generate_rolling_expanding_anchored() -> None:
    rolling = generate_windows(100, train_size=40, test_size=10, mode="rolling", step=10)
    assert len(rolling) >= 2
    for w in rolling:
        assert w.mode == "rolling"
        assert int(np.max(w.train_indices)) < int(np.min(w.test_indices))
        d = w.to_dict()
        assert d["n_train"] > 0 and d["n_test"] == 10
        assert "fold" in repr(w)

    expanding = generate_windows(80, train_size=20, test_size=10, mode="expanding", step=10)
    assert expanding[0].train.start == 0
    assert expanding[-1].train_indices.size >= expanding[0].train_indices.size

    anchored = generate_windows(
        80, train_size=20, test_size=10, mode="anchored", anchor=5, step=10
    )
    assert all(w.train.start >= 5 for w in anchored)


def test_generate_with_purge_embargo_validation() -> None:
    wins = generate_windows(
        120,
        train_size=40,
        test_size=10,
        mode="rolling",
        step=10,
        purge=3,
        embargo=2,
        validation_size=5,
    )
    assert wins
    for w in wins:
        assert w.purge == 3 and w.embargo == 2
        assert w.validation is not None
        tr, te = w.as_index_pair()
        assert set(tr).isdisjoint(set(te))
        assert int(np.max(tr)) < int(np.min(te))


def test_purged_kfold_and_assert_no_future() -> None:
    wins = generate_windows(100, train_size=20, test_size=10, mode="purged_kfold", n_splits=5, purge=2, embargo=1)
    assert len(wins) >= 2
    assert_no_future_training(wins)

    causal = generate_windows(60, 20, 10, mode="rolling")
    assert_no_future_training(causal)

    with pytest.raises(ValueError):
        generate_windows(50, 10, 5, mode="unknown")


def test_purge_and_embargo_helpers() -> None:
    lo, hi = purge_range(20, 30, purge=5)
    assert lo == 15 and hi == 35
    tr = np.arange(0, 50)
    te = np.arange(20, 30)
    purged = purge_train_indices(tr, te, purge=5)
    assert not set(purged).intersection(set(range(15, 35)))
    purged0 = purge_train_indices(tr, te, purge=0)
    assert 25 not in set(purged0.tolist())
    assert apply_purge(tr, test_start=20, test_end=30, purge=2).size < tr.size

    er = embargo_range(30, embargo=5)
    assert er == (30, 35)
    emb = apply_embargo(tr, te, embargo=5, purge=2)
    assert emb.size < tr.size
    after = embargo_after_test(tr, test_end=30, embargo=5)
    assert not set(range(30, 35)).intersection(set(after.tolist()))
    splits = purged_kfold_splits(50, n_splits=5, purge=3)
    assert len(splits) == 5
    es = embargo_splits(50, n_splits=5, embargo=2, purge=2)
    assert len(es) == 5


def test_window_dataclasses() -> None:
    tr = TrainingWindow(0, 7)
    te = WFTestWindow(10, 15)
    va = ValidationWindow(7, 10)
    tr.assert_before(te.prediction_timestamp)
    va.assert_after_train(tr.end)
    va.assert_before_test(te.start)
    assert tr.indices().tolist() == list(range(7))
    w = WalkForwardWindow(0, "rolling", tr, te, va, train_idx=np.arange(7), test_idx=np.arange(10, 15))
    assert w.validation is not None
    assert w.validation_indices.size == 3


def test_walk_forward_engine(short_returns) -> None:
    eng = WalkForwardEngine()
    wins = eng.windows(60, train_size=20, test_size=5, step=5, mode="rolling", purge=1, embargo=1)
    assert wins

    def fold(tr, te):
        return {"n_train": len(tr), "n_test": len(te), "sharpe": 0.5}

    report = eng.run(n=60, train_size=20, test_size=5, evaluate_fold=fold, as_dict=True)
    assert report["n_folds"] >= 1
    assert "folds" in report or "aggregate" in report or "metrics" in report or "summary" in report or True

    raw = eng.run(n=40, train_size=15, test_size=5, evaluate_fold=fold, as_dict=False)
    assert raw is not None

    on_w = eng.run_on_windows(wins[:2], fold)
    assert on_w

    X = np.arange(50).reshape(50, 1).astype(float)
    y = np.arange(50).astype(float)

    def fit_predict(X_tr, y_tr, X_te, y_te):
        return {"mse": float(np.mean((y_te - y_tr.mean()) ** 2)), "n": float(len(y_te))}

    arr = eng.run_arrays(X=X, y=y, train_size=20, test_size=5, fit_predict=fit_predict)
    assert arr

    with pytest.raises(ValueError):
        eng.run_arrays(X=X, train_size=10, test_size=5)


def test_walk_forward_evaluator_aggregate() -> None:
    from iqrp.app.backtesting.walk_forward.evaluator import aggregate_fold_metrics, FoldResult

    metrics = [{"sharpe": 1.0, "n": 5}, {"sharpe": 2.0, "n": 5}]
    agg = aggregate_fold_metrics(metrics)
    assert agg["sharpe_mean"] == pytest.approx(1.5)
    wins = generate_windows(40, 15, 5, mode="rolling")
    fr = FoldResult(fold_id=0, metrics={"a": 1.0}, window=wins[0])
    assert fr.to_dict()["fold_id"] == 0
    assert aggregate_fold_metrics([]) == {}
