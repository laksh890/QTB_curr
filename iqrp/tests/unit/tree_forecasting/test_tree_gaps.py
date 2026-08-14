"""Coverage gap tests for tree forecasting engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from iqrp.app.forecasting.tree_models import TreeSettings, TreeTrainer, create_tree_model
from iqrp.app.forecasting.tree_models.base.backends import (
    create_estimator,
    estimator_feature_importances,
    estimator_predict_proba,
)
from iqrp.app.forecasting.tree_models.base.ensemble import ensemble_fit_predict
from iqrp.app.forecasting.tree_models.base.native import NativeForest, NativeGBM, _Tree
from iqrp.app.forecasting.tree_models.base.processes import (
    feature_names,
    simulate_nonlinear_returns,
)
from iqrp.app.forecasting.tree_models.calibration.calibrators import (
    apply_calibration,
    fit_calibrator,
)
from iqrp.app.forecasting.tree_models.explainability.importance import compute_feature_importance
from iqrp.app.forecasting.tree_models.optimization.hpo import optimize_hyperparameters
from iqrp.app.forecasting.tree_models.registry import ensure_tree_models_loaded
from iqrp.app.forecasting.tree_models.visualization import plots as plot_mod


@pytest.mark.unit
def test_backend_fallbacks_and_importances() -> None:
    X = np.random.default_rng(0).normal(size=(80, 4))
    y = X[:, 0] + 0.1 * np.random.default_rng(1).normal(size=80)
    for backend in (
        "xgboost",
        "lightgbm",
        "catboost",
        "hist_gradient_boosting",
        "random_forest",
        "extra_trees",
    ):
        est = create_estimator(backend, task="regression", params={"n_estimators": 15, "max_depth": 2})  # type: ignore[arg-type]
        est.fit(X, y)
        imp = estimator_feature_importances(est, 4)
        assert imp.size == 4
    # classification proba
    yb = (y > 0).astype(float)
    est = create_estimator(
        "random_forest", task="binary", params={"n_estimators": 15, "max_depth": 2}
    )
    est.fit(X, yb)
    assert estimator_predict_proba(est, X).shape[1] >= 2
    # unknown backend
    with pytest.raises(Exception):
        create_estimator("nope", task="regression")  # type: ignore[arg-type]


@pytest.mark.unit
def test_native_tree_edges() -> None:
    X = np.random.default_rng(3).normal(size=(40, 3))
    y = np.ones(40)
    t = _Tree(max_depth=2, min_leaf=5, random_state=0, extra=True).fit(X, y)
    assert t.predict(X).size == 40
    nf = NativeForest(n_estimators=5, max_depth=2, task="classification", extra=True).fit(
        X, (X[:, 0] > 0).astype(float)
    )
    assert nf.predict_proba(X).shape[1] == 2
    gbm = NativeGBM(n_estimators=8, max_depth=2, task="classification", quantile_alpha=None).fit(
        X, (X[:, 0] > 0).astype(float)
    )
    assert gbm.predict_proba(X).shape[1] == 2
    gbm_q = NativeGBM(n_estimators=8, max_depth=2, quantile_alpha=0.5).fit(X, y)
    assert gbm_q.predict(X).size == 40


@pytest.mark.unit
def test_model_error_paths_and_online_modes() -> None:
    frame = simulate_nonlinear_returns(100, n_features=4, rng=np.random.default_rng(4))
    cols = feature_names(4)
    m = create_tree_model("random_forest")
    with pytest.raises(Exception):
        m.predict(frame)
    with pytest.raises(Exception):
        m.fit(frame.select(["open_time", "target"]))
    settings = TreeSettings.from_mapping(
        {"online": {"mode": "refit"}, "hyperparameters": {"n_estimators": 15, "max_depth": 2}}
    )
    m2 = create_tree_model("extra_trees", settings=settings)
    m2.fit(frame[:60], feature_columns=cols)
    m2.partial_fit(frame[60:], feature_columns=cols)
    settings3 = TreeSettings.from_mapping(
        {
            "online": {"mode": "incremental", "refresh_every": 1},
            "hyperparameters": {"n_estimators": 15, "max_depth": 2},
        }
    )
    m3 = create_tree_model("random_forest", settings=settings3)
    m3.fit(frame[:50], feature_columns=cols)
    m3.partial_fit(frame[50:70], feature_columns=cols)
    # settings none / dict
    from iqrp.app.forecasting.tree_models.sklearn.random_forest import RandomForestForecastModel

    m4 = RandomForestForecastModel(settings=None)
    m4.fit(frame, feature_columns=cols)
    m5 = RandomForestForecastModel(settings={"hyperparameters": {"n_estimators": 10}})
    m5.fit(frame, feature_columns=cols)


@pytest.mark.unit
def test_predict_proba_gate_and_explain_shap() -> None:
    frame = simulate_nonlinear_returns(
        120, n_features=4, classification=True, rng=np.random.default_rng(5)
    )
    cols = feature_names(4)
    # regression model still has supports_proba True on meta — use binary task
    settings = TreeSettings.from_mapping(
        {"task": {"type": "binary"}, "hyperparameters": {"n_estimators": 20, "max_depth": 2}}
    )
    m = create_tree_model("lightgbm", settings=settings)
    m.fit(frame, feature_columns=cols)
    expl = m.explain(frame, method="shap")
    assert expl.method == "shap"
    expl2 = m.explain(frame, method="permutation")
    assert expl2.importances
    # feature importance kinds
    assert m.feature_importance(kind="split")
    assert m.feature_importance(kind="permutation")
    assert m.feature_importance(kind="shap")


@pytest.mark.unit
def test_calibration_none_and_apply_1d() -> None:
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    p = np.array([0.2, 0.7, 0.8, 0.3, 0.6])
    assert fit_calibrator(y, p, method="none") is None
    cal = fit_calibrator(y, p, method="platt")
    out = apply_calibration(cal, p)
    assert out.ndim == 1
    out2 = apply_calibration(None, p)
    assert np.allclose(out2, p)
    # multiclass-ish matrix
    P = np.column_stack([1 - p, p * 0.5, p * 0.5])
    cal2 = fit_calibrator(y, P, method="temperature")
    assert apply_calibration(cal2, P).shape == P.shape


@pytest.mark.unit
def test_hpo_parallel_and_bayesian() -> None:
    X = np.random.default_rng(6).normal(size=(120, 3))
    y = X.sum(axis=1)
    from iqrp.app.forecasting.tree_models.config import ValidationConfig

    best, scores = optimize_hyperparameters(
        "extra_trees",
        X,
        y,
        task="regression",
        base_params={"n_estimators": 15, "max_depth": 3},
        method="bayesian",
        n_trials=3,
        validation=ValidationConfig(train_size=50, test_size=15),
        parallel=True,
    )
    assert isinstance(best, dict)
    best2, _ = optimize_hyperparameters(
        "random_forest", X, y, task="regression", base_params={"n_estimators": 10}, method="none"
    )
    assert best2["n_estimators"] == 10


@pytest.mark.unit
def test_ensemble_stacking_blending() -> None:
    X = np.random.default_rng(7).normal(size=(100, 3))
    y = X[:, 0] - X[:, 1]
    for method in ("bagging", "average", "stacking", "blending"):
        pred = ensemble_fit_predict(
            ["random_forest", "extra_trees"],
            X,
            y,
            X[:10],
            method=method,  # type: ignore[arg-type]
            params={"n_estimators": 12, "max_depth": 2},
            n_bags=2,
        )
        assert pred.size == 10


@pytest.mark.unit
def test_trainer_parallel_failures_and_registry() -> None:
    frame = simulate_nonlinear_returns(80, n_features=3, rng=np.random.default_rng(8))
    trainer = TreeTrainer(
        TreeSettings.from_mapping(
            {
                "hyperparameters": {"n_estimators": 12, "max_depth": 2},
                "visualization": {"enabled": False},
            }
        )
    )
    rows = trainer.compare(
        ["random_forest", "not_a_model"],
        frame,
        feature_columns=feature_names(3),
        parallel=True,
    )
    assert len(rows) >= 1
    assert ensure_tree_models_loaded(["iqrp.app.forecasting.tree_models.nope"]) == []


@pytest.mark.unit
def test_plots_with_matplotlib() -> None:
    # matplotlib is installed
    assert plot_mod._pyplot() is not None
    payload = plot_mod.plot_feature_importance({"a": 1.0, "b": 0.2})
    assert "figure" in payload
    payload2 = plot_mod.plot_dependence(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    assert "figure" in payload2


@pytest.mark.unit
def test_config_default_missing(tmp_path, monkeypatch) -> None:
    from iqrp.app.forecasting.tree_models import config as cfg

    monkeypatch.setattr(cfg, "_default_config_path", lambda: tmp_path / "missing.yaml")
    assert cfg.TreeSettings.default().task.type == "regression"
    from omegaconf import OmegaConf

    s = cfg.TreeSettings.from_mapping(OmegaConf.create({"hyperparameters": {"n_estimators": 11}}))
    assert s.hyperparameters.n_estimators == 11


@pytest.mark.unit
def test_compute_importance_and_missing_cols() -> None:
    frame = simulate_nonlinear_returns(60, n_features=3, rng=np.random.default_rng(9))
    cols = feature_names(3)
    m = create_tree_model(
        "random_forest",
        settings=TreeSettings.from_mapping(
            {"hyperparameters": {"n_estimators": 10, "max_depth": 2}}
        ),
    )
    m.fit(frame, feature_columns=cols)
    assert compute_feature_importance(m._estimator, cols, kind="gain")
    with pytest.raises(Exception):
        m.predict(frame.select(["open_time", "target", "f0"]))  # missing f1,f2


@pytest.mark.unit
def test_quantile_xgboost_and_hist() -> None:
    frame = simulate_nonlinear_returns(100, n_features=3, rng=np.random.default_rng(10))
    cols = feature_names(3)
    settings = TreeSettings.from_mapping(
        {
            "task": {"type": "quantile", "quantile_alphas": [0.5]},
            "hyperparameters": {"n_estimators": 15, "max_depth": 2},
        }
    )
    for name in ("xgboost", "catboost", "hist_gradient_boosting"):
        m = create_tree_model(name, settings=settings)
        m.fit(frame, feature_columns=cols)
        assert m.predict(frame).size == frame.height
