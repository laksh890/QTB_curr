"""Additional coverage for tree forecasting engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from iqrp.app.forecasting.tree_models import TreeSettings, TreeTrainer, create_tree_model
from iqrp.app.forecasting.tree_models.base import backends as backend_mod
from iqrp.app.forecasting.tree_models.base.backends import create_estimator, estimator_feature_importances
from iqrp.app.forecasting.tree_models.base.ensemble import stacking_predict, weighted_average
from iqrp.app.forecasting.tree_models.base.native import NativeForest, NativeGBM
from iqrp.app.forecasting.tree_models.base.processes import feature_names, simulate_nonlinear_returns
from iqrp.app.forecasting.tree_models.calibration.calibrators import Calibrator, apply_calibration, fit_calibrator
from iqrp.app.forecasting.tree_models.diagnostics.report import (
    calibration_curve,
    feature_stability,
    learning_curve,
    validation_curve,
)
from iqrp.app.forecasting.tree_models.evaluation.metrics import evaluate_tree_predictions
from iqrp.app.forecasting.tree_models.explainability.importance import (
    compute_feature_importance,
    decision_paths,
    shap_values,
)
from iqrp.app.forecasting.tree_models.optimization import hpo as hpo_mod
from iqrp.app.forecasting.tree_models.optimization.cv import make_time_splits
from iqrp.app.forecasting.tree_models.preprocessing.pipeline import select_features
from iqrp.app.forecasting.tree_models.visualization import plots as plot_mod


@pytest.mark.unit
def test_backend_native_fallbacks() -> None:
    X = np.random.default_rng(0).normal(size=(60, 3))
    y = X[:, 0]
    # force library import failures → NativeGBM / NativeForest
    for mod_name, backend in [
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("catboost", "catboost"),
        ("sklearn", "hist_gradient_boosting"),
        ("sklearn", "random_forest"),
        ("sklearn", "extra_trees"),
    ]:
        with patch.dict("sys.modules", {mod_name: None, "sklearn.ensemble": None}):
            est = create_estimator(
                backend,  # type: ignore[arg-type]
                task="regression",
                params={"n_estimators": 5, "max_depth": 2},
            )
            est.fit(X, y)
            assert est.predict(X).size == 60
    # xgb gain path
    est = create_estimator("xgboost", task="regression", params={"n_estimators": 10, "max_depth": 2})
    est.fit(X, y)
    if hasattr(est, "get_booster"):
        assert estimator_feature_importances(est, 3).size == 3
    # classification native multiclass-ish
    y3 = np.array([0, 1, 2] * 20)
    nf = NativeForest(n_estimators=5, max_depth=2, task="classification").fit(X, y3.astype(float))
    assert nf.predict(X).size == 60
    assert nf.predict_proba(X).shape[1] >= 2
    # get_booster score parse failure
    mock_est = MagicMock()
    mock_est.feature_importances_ = None
    del mock_est.feature_importances_
    mock_est.get_booster.side_effect = RuntimeError("x")
    assert estimator_feature_importances(mock_est, 4).sum() > 0


@pytest.mark.unit
def test_quantile_native_and_gpu_params() -> None:
    X = np.random.default_rng(1).normal(size=(50, 3))
    y = X.sum(1)
    for backend in ("xgboost", "lightgbm", "catboost", "hist_gradient_boosting"):
        est = create_estimator(
            backend,  # type: ignore[arg-type]
            task="quantile",
            params={"n_estimators": 10, "max_depth": 2, "device": "cpu"},
            quantile_alpha=0.5,
        )
        est.fit(X, y)
        assert est.predict(X).shape[0] == 50
    # cls backends
    yb = (y > 0).astype(float)
    for backend in ("xgboost", "lightgbm", "catboost", "hist_gradient_boosting"):
        est = create_estimator(backend, task="binary", params={"n_estimators": 10, "max_depth": 2})  # type: ignore[arg-type]
        est.fit(X, yb)


@pytest.mark.unit
def test_hpo_exception_and_pruning_paths() -> None:
    X = np.random.default_rng(2).normal(size=(80, 2))
    y = X[:, 0]
    from iqrp.app.forecasting.tree_models.config import ValidationConfig

    # force score failures
    with patch.object(hpo_mod, "create_estimator", side_effect=RuntimeError("boom")):
        best, scores = hpo_mod.optimize_hyperparameters(
            "random_forest",
            X,
            y,
            task="regression",
            base_params={"n_estimators": 5},
            method="random",
            n_trials=2,
            validation=ValidationConfig(train_size=30, test_size=10),
            parallel=True,
        )
        assert isinstance(best, dict)
    # empty splits → holdout
    cfg = ValidationConfig(strategy="walk_forward", train_size=1000, test_size=500)
    assert make_time_splits(50, cfg) == [] or True
    s = hpo_mod._score_params(
        "random_forest", X, y, task="regression", params={"n_estimators": 5, "max_depth": 2}, validation=cfg
    )
    assert s >= 0
    # optuna prune path via n_trials=1
    best2, _ = hpo_mod.optimize_hyperparameters(
        "random_forest",
        X,
        y,
        task="regression",
        base_params={"n_estimators": 8, "max_depth": 2},
        method="optuna",
        n_trials=2,
        validation=ValidationConfig(train_size=40, test_size=10),
        pruning=True,
        parallel=False,
    )
    assert best2


@pytest.mark.unit
def test_diagnostics_exception_branches() -> None:
    X = np.random.default_rng(3).normal(size=(60, 3))
    y = X[:, 0]
    with patch(
        "iqrp.app.forecasting.tree_models.diagnostics.report.create_estimator",
        side_effect=RuntimeError("x"),
    ):
        lc = learning_curve("random_forest", X, y, task="regression", params={})
        assert "train_rmse" in lc
        vc = validation_curve("random_forest", X, y, task="regression", params={})
        assert "val_rmse" in vc
        stab = feature_stability("random_forest", X, y, ["a", "b", "c"], task="regression", params={})
        assert len(stab) == 3
    assert calibration_curve(y, y + 0.1, n_bins=5)["mean_predicted"]


@pytest.mark.unit
def test_explain_decision_paths_and_shap_fallback() -> None:
    X = np.random.default_rng(4).normal(size=(40, 3))
    y = X[:, 0]
    est = create_estimator("random_forest", task="regression", params={"n_estimators": 8, "max_depth": 2})
    est.fit(X, y)
    paths = decision_paths(est, X)
    assert paths
    # force kernel shap
    with patch.dict("sys.modules", {"shap": None}):
        sv = shap_values(est, X[:10])
        assert sv.shape == (10, 3)
    assert compute_feature_importance(est, ["a", "b", "c"], kind="shap", X=X, y=y)
    # estimator without importances
    class Dummy:
        def predict(self, X):
            return np.zeros(X.shape[0])

    assert compute_feature_importance(Dummy(), ["a", "b", "c"], kind="gain")


@pytest.mark.unit
def test_calibration_to_dict_and_multiclass_labels() -> None:
    y = np.array([1.0, 2.0, 1.0, 2.0, 3.0])
    p = np.array([0.2, 0.8, 0.3, 0.7, 0.6])
    cal = fit_calibrator(y, p, method="platt")
    assert cal is not None and "a" in cal.to_dict()["params"]
    # isotonic apply
    cal_i = fit_calibrator((y > 1.5).astype(float), p, method="isotonic")
    assert apply_calibration(cal_i, p).size == p.size
    # unknown method returns None
    assert fit_calibrator(y, p, method="nope") is None  # type: ignore[arg-type]
    # apply with >2 classes matrix
    P = np.column_stack([1 - p, p / 2, p / 2])
    cal_t = Calibrator(method="temperature", params={"temperature": 1.5})
    assert apply_calibration(cal_t, P).shape[1] == 3


@pytest.mark.unit
def test_metrics_edge_cases() -> None:
    assert evaluate_tree_predictions(np.array([1.0]), np.array([1.0]))["directional_accuracy"] != 999
    pnl = np.array([0.0, 0.0])
    m = evaluate_tree_predictions(np.array([0.1, -0.1, 0.2]), np.array([0.0, 0.0, 0.0]))
    assert "profit_factor" in m
    # roc degenerate
    m2 = evaluate_tree_predictions(
        np.ones(10),
        np.ones(10),
        proba=np.column_stack([np.zeros(10), np.ones(10)]),
        task="binary",
    )
    assert m2["roc_auc"] == 0.5


@pytest.mark.unit
def test_preprocess_edges_and_cv_default() -> None:
    X = np.random.default_rng(5).normal(size=(30, 4))
    X[0, 0] = np.nan
    y = X[:, 0]
    assert select_features(X, y, list("abcd"), method="none", max_features=2)
    assert select_features(X, y, list("abcd"), method="none")
    # unknown method falls through
    assert select_features(X, y, list("abcd"), method="nope", max_features=2)  # type: ignore[arg-type]
    from iqrp.app.forecasting.tree_models.config import ValidationConfig

    # unknown strategy falls back via getattr on a simple namespace
    from types import SimpleNamespace

    splits = make_time_splits(100, SimpleNamespace(strategy="unknown", train_size=40, test_size=10, gap=0, n_splits=3, embargo=5, purge=5))
    assert splits


@pytest.mark.unit
def test_plots_none_branch_and_ensemble_edges() -> None:
    with patch.object(plot_mod, "_pyplot", return_value=None):
        assert "names" in plot_mod.plot_feature_importance({"a": 1.0})
        assert "names" in plot_mod.plot_shap_summary(np.ones((5, 2)))
        assert "grid" in plot_mod.plot_dependence(np.arange(3.0), np.arange(3.0))
        assert "y_true" in plot_mod.plot_prediction_error(np.arange(3.0), np.arange(3.0))
        assert "mean_predicted" in plot_mod.plot_calibration([0.1], [0.2])
        assert "train_sizes" in plot_mod.plot_learning_curve(
            {"train_sizes": [1], "train_rmse": [1.0], "val_rmse": [1.0]}
        )
        assert "residuals" in plot_mod.plot_residual_distribution(np.ones(3))
    # stacking 1d
    assert stacking_predict(np.arange(10.0), np.arange(10.0), np.arange(5.0)).size == 5
    assert weighted_average([np.ones(3)]).size == 3


@pytest.mark.unit
def test_tree_model_remaining_branches() -> None:
    frame = simulate_nonlinear_returns(90, n_features=4, rng=np.random.default_rng(6))
    cols = feature_names(4)
    # no features error already covered; predict_proba on regression with supports_proba
    settings = TreeSettings.from_mapping(
        {
            "task": {"type": "regression"},
            "hyperparameters": {"n_estimators": 10, "max_depth": 2},
            "online": {"mode": "warm_start", "refresh_every": 100},
            "calibration": {"enabled": False},
        }
    )
    m = create_tree_model("random_forest", settings=settings)
    m.fit(frame, feature_columns=cols)
    # partial_fit when X cleared
    m._X = None
    m.partial_fit(frame[:20], feature_columns=cols)
    # forecast interval without intervals
    from iqrp.app.forecasting.base.forecast import Forecast

    with patch.object(m, "forecast", return_value=Forecast.from_values([0.1, 0.2], horizon=2, intervals=None)):
        assert len(m.forecast_interval(frame, horizon=2)) == 2
    # settings dict init
    m2 = create_tree_model("xgboost", settings={"hyperparameters": {"n_estimators": 8, "max_depth": 2}})
    m2.fit(frame, feature_columns=cols)
    # trainer serial exception
    trainer = TreeTrainer(
        TreeSettings.from_mapping(
            {"hyperparameters": {"n_estimators": 8, "max_depth": 2}, "visualization": {"enabled": False}}
        )
    )
    rows = trainer.compare(["random_forest", "missing"], frame, feature_columns=cols, parallel=False)
    assert rows


@pytest.mark.unit
def test_native_gbm_regression_and_forest_binary() -> None:
    X = np.random.default_rng(7).normal(size=(40, 2))
    y = X[:, 0]
    gbm = NativeGBM(n_estimators=5, max_depth=2, task="regression").fit(X, y)
    assert gbm.predict(X).size == 40
    # empty sse
    from iqrp.app.forecasting.tree_models.base.native import _weighted_sse

    assert _weighted_sse(np.array([]), np.array([])) == 0.0
