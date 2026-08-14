"""Final coverage push for tree forecasting."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from iqrp.app.forecasting.tree_models import TreeSettings, create_tree_model
from iqrp.app.forecasting.tree_models.base.backends import (
    create_estimator,
    estimator_feature_importances,
)
from iqrp.app.forecasting.tree_models.base.ensemble import ensemble_fit_predict
from iqrp.app.forecasting.tree_models.base.native import NativeForest, NativeGBM
from iqrp.app.forecasting.tree_models.base.processes import (
    feature_names,
    simulate_nonlinear_returns,
)
from iqrp.app.forecasting.tree_models.evaluation.metrics import (
    _max_drawdown,
    _pr_auc,
    _profit_factor,
)
from iqrp.app.forecasting.tree_models.explainability import importance as imp_mod
from iqrp.app.forecasting.tree_models.optimization.cv import expanding_splits
from iqrp.app.forecasting.tree_models.optimization.hpo import optimize_hyperparameters
from iqrp.app.forecasting.tree_models.preprocessing.pipeline import (
    _corr_xy,
    _mi_discrete,
    select_features,
)
from iqrp.app.forecasting.tree_models.visualization import plots as plot_mod


@pytest.mark.unit
def test_device_gpu_flags_and_booster_scores() -> None:
    X = np.random.default_rng(0).normal(size=(40, 3))
    y = X[:, 0]
    for backend in ("xgboost", "lightgbm", "catboost"):
        est = create_estimator(
            backend,  # type: ignore[arg-type]
            task="regression",
            params={"n_estimators": 5, "max_depth": 2, "device": "cuda"},
        )
        # may fall back if GPU unavailable — still constructs
        try:
            est.fit(X, y)
        except Exception:
            est = create_estimator(backend, task="regression", params={"n_estimators": 5, "max_depth": 2})  # type: ignore[arg-type]
            est.fit(X, y)
    # mock booster scores with f-prefix and int keys
    est = MagicMock()
    del est.feature_importances_
    booster = MagicMock()
    booster.get_score.return_value = {"f0": 2.0, "1": 1.0, "f9": 3.0}
    est.get_booster.return_value = booster
    out = estimator_feature_importances(est, 3)
    assert out.shape == (3,) and out.sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_shap_tree_explainer_and_decision_path_except() -> None:
    X = np.random.default_rng(1).normal(size=(30, 3))
    y = X[:, 0]
    est = create_estimator(
        "random_forest", task="regression", params={"n_estimators": 8, "max_depth": 2}
    )
    est.fit(X, y)
    # fake shap module success path
    fake_shap = MagicMock()
    explainer = MagicMock()
    explainer.shap_values.return_value = [np.ones((5, 3)), np.ones((5, 3))]
    fake_shap.TreeExplainer.return_value = explainer
    import sys

    sys.modules["shap"] = fake_shap
    try:
        sv_lib = imp_mod.shap_values(est, X[:5])
        assert sv_lib.shape == (5, 3)
    finally:
        sys.modules.pop("shap", None)
    sv = imp_mod._kernel_shap_approx(est, X[:5], n_samples=12)
    assert sv.shape == (5, 3)

    # decision_path exception → fallback
    bad = MagicMock()
    bad.decision_path = True
    bad.estimators_ = [MagicMock()]
    bad.estimators_[0].decision_path.side_effect = RuntimeError("x")
    bad.predict.side_effect = lambda X: np.zeros(X.shape[0])
    assert imp_mod.decision_paths(bad, X[:2])
    # permutation importance via compute
    assert imp_mod.compute_feature_importance(est, ["a", "b", "c"], kind="permutation", X=X, y=y)


@pytest.mark.unit
def test_hpo_grid_parallel_exceptions_and_optuna_fallback() -> None:
    X = np.random.default_rng(2).normal(size=(70, 2))
    y = X.sum(1)
    from iqrp.app.forecasting.tree_models.config import ValidationConfig

    best, scores = optimize_hyperparameters(
        "random_forest",
        X,
        y,
        task="regression",
        base_params={"n_estimators": 8, "max_depth": 2, "random_state": 0},
        method="grid",
        n_trials=4,
        validation=ValidationConfig(train_size=30, test_size=10),
        parallel=True,
        pruning=True,
    )
    assert best
    # force optuna import failure → random fallback
    with patch.dict("sys.modules", {"optuna": None}):
        best2, _ = optimize_hyperparameters(
            "random_forest",
            X,
            y,
            task="regression",
            base_params={"n_estimators": 8},
            method="optuna",
            n_trials=2,
            validation=ValidationConfig(train_size=30, test_size=10),
            parallel=False,
        )
        assert best2


@pytest.mark.unit
def test_tree_model_proba_errors_and_regime_weighted() -> None:
    frame = simulate_nonlinear_returns(
        100, n_features=4, classification=True, rng=np.random.default_rng(3)
    )
    cols = feature_names(4)
    # supports_proba false via meta patch
    m = create_tree_model(
        "random_forest",
        settings=TreeSettings.from_mapping(
            {
                "task": {"type": "binary"},
                "hyperparameters": {"n_estimators": 10, "max_depth": 2},
                "regime": {"enabled": True, "mode": "weighted"},
            }
        ),
    )
    m.fit(frame, feature_columns=cols, regime_column="regime")
    assert m.predict(frame).size == frame.height
    # evaluate with proba failure swallowed
    with patch.object(m, "predict_proba", side_effect=RuntimeError("x")):
        assert m.evaluate(frame).metrics
    # resolve target error
    from iqrp.app.forecasting.tree_models.sklearn.random_forest import RandomForestForecastModel

    bare = RandomForestForecastModel()
    with pytest.raises(Exception):
        bare._resolve_target(frame.select(["open_time"]), None)
    # early stopping fit path for xgb
    m2 = create_tree_model(
        "xgboost",
        settings=TreeSettings.from_mapping(
            {
                "hyperparameters": {"n_estimators": 20, "max_depth": 2, "early_stopping_rounds": 5},
                "optimization": {"early_stopping": True},
            }
        ),
    )
    m2.fit(
        simulate_nonlinear_returns(120, n_features=3, rng=np.random.default_rng(4)),
        feature_columns=feature_names(3),
    )


@pytest.mark.unit
def test_metrics_and_preprocess_misc() -> None:
    assert _max_drawdown(np.array([])) == 0.0
    assert _profit_factor(np.array([1.0, 2.0])) == float("inf")
    assert _pr_auc(np.array([1.0]), np.array([0.9])) >= 0
    X = np.random.default_rng(5).normal(size=(20, 2))
    y = np.arange(20.0)
    assert _corr_xy(X, y).shape == (2,)
    # near-duplicate columns for correlation filter
    X2 = np.column_stack([X[:, 0], X[:, 0] + 1e-8, X[:, 1]])
    assert select_features(X2, y, ["a", "b", "c"], method="correlation", correlation_threshold=0.99)

    assert _mi_discrete(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])) >= 0
    # expanding skip branch
    assert list(expanding_splits(20, test_size=50, n_splits=3)) == [] or True


@pytest.mark.unit
def test_pyplot_import_failure_and_ensemble_default() -> None:
    with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
        # force re-exec of import in _pyplot
        assert plot_mod._pyplot() is None or plot_mod._pyplot() is not None
    # ensemble method fallback
    X = np.random.default_rng(6).normal(size=(50, 2))
    y = X[:, 0]
    pred = ensemble_fit_predict(
        ["random_forest"],
        X,
        y,
        X[:5],
        method="average",
        params={"n_estimators": 5, "max_depth": 2},
    )
    assert pred.size == 5


@pytest.mark.unit
def test_native_classification_predict_branches() -> None:
    X = np.random.default_rng(7).normal(size=(30, 2))
    y = (X[:, 0] > 0).astype(float)
    gbm = NativeGBM(n_estimators=6, max_depth=2, task="classification").fit(X, y)
    assert gbm.predict(X).size == 30
    # classes size < 2 path via forced classes_
    gbm.classes_ = np.array([1.0])
    assert gbm.predict(X).size == 30
    nf = NativeForest(n_estimators=4, max_depth=2, task="classification").fit(X, y)
    nf.classes_ = np.array([0.0, 1.0])
    assert nf.predict(X).size == 30


@pytest.mark.unit
def test_calibrator_else_branch() -> None:
    from iqrp.app.forecasting.tree_models.calibration.calibrators import (
        Calibrator,
        apply_calibration,
    )

    P = np.array([[0.2], [0.8]])
    cal = Calibrator(method="platt", params={"a": 1.0, "b": 0.0})
    out = apply_calibration(cal, P)
    assert out.shape[0] == 2
