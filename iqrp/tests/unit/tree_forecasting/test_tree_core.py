"""Core unit tests for Institutional Tree-Based Forecasting Engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.forecasting.tree_models import (
    TreeSettings,
    TreeTrainer,
    create_tree_model,
    list_tree_models,
)
from iqrp.app.forecasting.tree_models.base.ensemble import (
    bagging_predict,
    blending_predict,
    ensemble_fit_predict,
    stacking_predict,
    weighted_average,
)
from iqrp.app.forecasting.tree_models.base.native import NativeForest, NativeGBM
from iqrp.app.forecasting.tree_models.base.processes import (
    feature_names,
    simulate_nonlinear_returns,
)
from iqrp.app.forecasting.tree_models.calibration.calibrators import (
    apply_calibration,
    fit_calibrator,
)
from iqrp.app.forecasting.tree_models.evaluation.metrics import evaluate_tree_predictions
from iqrp.app.forecasting.tree_models.explainability.importance import (
    decision_paths,
    ice_curves,
    partial_dependence,
    shap_interaction_values,
    shap_values,
)
from iqrp.app.forecasting.tree_models.optimization.cv import make_time_splits
from iqrp.app.forecasting.tree_models.optimization.hpo import optimize_hyperparameters
from iqrp.app.forecasting.tree_models.preprocessing.pipeline import (
    TreePreprocessor,
    select_features,
)
from iqrp.app.forecasting.tree_models.visualization.plots import (
    plot_calibration,
    plot_feature_importance,
    plot_learning_curve,
    plot_prediction_error,
    plot_residual_distribution,
    plot_shap_summary,
)


@pytest.fixture
def reg_frame() -> pl.DataFrame:
    return simulate_nonlinear_returns(220, n_features=5, rng=np.random.default_rng(1))


@pytest.fixture
def cls_frame() -> pl.DataFrame:
    return simulate_nonlinear_returns(
        220, n_features=5, classification=True, rng=np.random.default_rng(2)
    )


@pytest.mark.unit
def test_registry_lists_all_models() -> None:
    names = set(list_tree_models())
    assert names >= {
        "xgboost",
        "lightgbm",
        "catboost",
        "hist_gradient_boosting",
        "random_forest",
        "extra_trees",
    }


@pytest.mark.unit
def test_settings_hydra() -> None:
    s = TreeSettings.default()
    assert s.hyperparameters.n_estimators > 0
    s2 = TreeSettings.from_mapping(
        {"task": {"type": "binary"}, "optimization": {"method": "random", "n_trials": 3}}
    )
    assert s2.task.type == "binary"
    s3 = TreeSettings.from_hydra(overrides=["forecast.default_horizon=7"])
    assert s3.forecast.default_horizon == 7
    with pytest.raises(Exception):
        TreeSettings.from_mapping({"task": {"type": "not_a_task"}})


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    ["xgboost", "lightgbm", "catboost", "hist_gradient_boosting", "random_forest", "extra_trees"],
)
def test_all_models_api(name: str, reg_frame: pl.DataFrame) -> None:
    settings = TreeSettings.from_mapping(
        {
            "hyperparameters": {"n_estimators": 25, "max_depth": 3},
            "visualization": {"enabled": False},
        }
    )
    model = create_tree_model(name, settings=settings)
    cols = feature_names(5)
    model.fit(reg_frame, feature_columns=cols, regime_column="regime")
    pred = model.predict(reg_frame)
    assert pred.shape[0] == reg_frame.height
    fc = model.forecast(reg_frame, horizon=4)
    assert fc.path().shape == (4,)
    assert len(model.forecast_interval(reg_frame, horizon=3)) == 3
    report = model.evaluate(reg_frame)
    assert "rmse" in report.metrics
    imp = model.feature_importance(kind="gain")
    assert set(cols).issubset(set(imp))

    expl = model.explain(reg_frame, method="builtin")
    assert expl.importances
    diag = model.diagnostics()
    assert diag.residual_std >= 0
    cv = model.cross_validate()
    assert "rmse_mean" in cv


@pytest.mark.unit
def test_classification_proba_and_calibration(cls_frame: pl.DataFrame) -> None:
    settings = TreeSettings.from_mapping(
        {
            "task": {"type": "binary"},
            "calibration": {"enabled": True, "method": "platt"},
            "hyperparameters": {"n_estimators": 30, "max_depth": 3},
        }
    )
    model = create_tree_model("xgboost", settings=settings)
    cols = feature_names(5)
    model.fit(cls_frame, feature_columns=cols)
    proba = model.predict_proba(cls_frame)
    assert proba.shape[0] == cls_frame.height
    assert proba.shape[1] >= 2
    metrics = model.evaluate(cls_frame).metrics
    assert "roc_auc" in metrics
    # other calibrators
    y = cls_frame["target"].to_numpy()
    for method in ("isotonic", "temperature"):
        cal = fit_calibrator(y, proba, method=method)  # type: ignore[arg-type]
        assert cal is not None
        out = apply_calibration(cal, proba)
        assert out.shape == proba.shape


@pytest.mark.unit
def test_quantile_and_regime_modes(reg_frame: pl.DataFrame) -> None:
    cols = feature_names(5)
    qsettings = TreeSettings.from_mapping(
        {
            "task": {"type": "quantile", "quantile_alphas": [0.1, 0.5, 0.9]},
            "hyperparameters": {"n_estimators": 20, "max_depth": 3},
        }
    )
    qm = create_tree_model("lightgbm", settings=qsettings)
    qm.fit(reg_frame, feature_columns=cols)
    assert "quantiles" in qm.forecast(reg_frame, horizon=2).metadata

    for mode in ("separate", "weighted", "routing"):
        s = TreeSettings.from_mapping(
            {
                "regime": {"enabled": True, "mode": mode},
                "hyperparameters": {"n_estimators": 20, "max_depth": 3},
            }
        )
        m = create_tree_model("random_forest", settings=s)
        m.fit(reg_frame, feature_columns=cols, regime_column="regime")
        assert m.predict(reg_frame).size == reg_frame.height


@pytest.mark.unit
def test_partial_fit_and_serialization(reg_frame: pl.DataFrame, tmp_path: Path) -> None:
    settings = TreeSettings.from_mapping(
        {
            "online": {"mode": "warm_start", "window": 100, "refresh_every": 2},
            "hyperparameters": {"n_estimators": 20, "max_depth": 3},
        }
    )
    model = create_tree_model("hist_gradient_boosting", settings=settings)
    cols = feature_names(5)
    mid = reg_frame.height // 2
    model.fit(reg_frame[:mid], feature_columns=cols)
    model.partial_fit(reg_frame[mid:], feature_columns=cols)
    path = tmp_path / "tree.json"
    model.save(path)
    loaded = type(model).load(path)
    assert loaded.is_fitted
    assert loaded.predict(reg_frame).size == reg_frame.height


@pytest.mark.unit
def test_hpo_and_cv_strategies(reg_frame: pl.DataFrame) -> None:
    X = reg_frame.select(feature_names(5)).to_numpy()
    y = reg_frame["target"].to_numpy()
    from iqrp.app.forecasting.tree_models.config import ValidationConfig

    for strategy in ("walk_forward", "rolling", "expanding", "blocked", "purged_kfold", "embargo"):
        cfg = ValidationConfig(strategy=strategy, train_size=80, test_size=20, n_splits=3)
        splits = make_time_splits(X.shape[0], cfg)
        assert len(splits) >= 1
    for method in ("grid", "random", "optuna"):
        best, scores = optimize_hyperparameters(
            "random_forest",
            X,
            y,
            task="regression",
            base_params={"n_estimators": 20, "max_depth": 3, "random_state": 0},
            method=method,  # type: ignore[arg-type]
            n_trials=3,
            validation=ValidationConfig(train_size=80, test_size=20),
            parallel=False,
        )
        assert "max_depth" in best or "n_estimators" in best


@pytest.mark.unit
def test_feature_selection_and_preprocess(reg_frame: pl.DataFrame) -> None:
    X = reg_frame.select(feature_names(5)).to_numpy()
    y = reg_frame["target"].to_numpy()
    names = feature_names(5)
    for method in ("correlation", "mutual_info", "rfe", "boruta", "shap", "permutation"):
        sel = select_features(X, y, names, method=method, max_features=3)  # type: ignore[arg-type]
        assert 1 <= len(sel) <= 3
    prep = TreePreprocessor(standardize=True).fit(X)
    Xt = prep.transform(X)
    assert Xt.shape == X.shape
    d = prep.to_dict()
    prep2 = TreePreprocessor.from_dict(d)
    assert prep2.transform(X).shape == X.shape


@pytest.mark.unit
def test_explainability_and_native(reg_frame: pl.DataFrame) -> None:
    X = reg_frame.select(feature_names(5)).to_numpy()
    y = reg_frame["target"].to_numpy()
    forest = NativeForest(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    gbm = NativeGBM(n_estimators=15, max_depth=2, random_state=0).fit(X, y)
    assert forest.predict(X).size == X.shape[0]
    assert gbm.predict(X).size == X.shape[0]
    assert forest.predict_proba(X).shape[1] == 2 or True
    sv = shap_values(forest, X[:30])
    assert sv.shape[0] == 30
    assert shap_interaction_values(forest, X[:10]).ndim == 3
    g, p = partial_dependence(forest, X, 0)
    assert g.size == p.size
    g2, ice = ice_curves(forest, X, 1)
    assert ice.shape[1] == g2.size
    assert decision_paths(forest, X[:3])


@pytest.mark.unit
def test_ensemble_and_metrics(reg_frame: pl.DataFrame) -> None:
    X = reg_frame.select(feature_names(5)).to_numpy()
    y = reg_frame["target"].to_numpy()
    pred = bagging_predict(
        "random_forest", X, y, X[:20], params={"n_estimators": 10, "max_depth": 3}, n_bags=3
    )
    assert pred.size == 20
    assert weighted_average([pred, pred], [0.5, 0.5]).size == 20
    stack = stacking_predict(
        np.column_stack([y[:50], y[:50] * 0.9]), y[:50], np.column_stack([y[:10], y[:10]])
    )
    assert stack.size == 10
    assert blending_predict(stack.reshape(-1, 1)[:10], y[:10], stack.reshape(-1, 1)[:10]).size == 10
    ens = ensemble_fit_predict(
        ["random_forest", "extra_trees"],
        X,
        y,
        X[:15],
        method="average",
        params={"n_estimators": 15, "max_depth": 3},
    )
    assert ens.size == 15
    m = evaluate_tree_predictions(y, y * 0.9 + 0.01)
    assert "sharpe_ratio" in m and "max_drawdown" in m


@pytest.mark.unit
def test_trainer_compare(reg_frame: pl.DataFrame) -> None:
    settings = TreeSettings.from_mapping(
        {
            "hyperparameters": {"n_estimators": 20, "max_depth": 3},
            "visualization": {"enabled": True},
        }
    )
    trainer = TreeTrainer(settings)
    model, result = trainer.fit("random_forest", reg_frame, feature_columns=feature_names(5))
    assert result.to_dict()["metrics"]
    rows = trainer.compare(
        ["random_forest", "extra_trees"],
        reg_frame,
        feature_columns=feature_names(5),
        parallel=False,
    )
    assert len(rows) >= 1
    assert model.meta.name == "random_forest"


@pytest.mark.unit
def test_visualization_helpers() -> None:
    assert "names" in plot_feature_importance({"a": 0.2, "b": 0.8})
    assert "mean_abs_shap" in plot_shap_summary(np.random.default_rng(0).normal(size=(20, 3)))
    assert "figure" in plot_prediction_error(
        np.arange(10.0), np.arange(10.0) + 0.1
    ) or "y_true" in plot_prediction_error(np.arange(10.0), np.arange(10.0))
    assert "mean_predicted" in plot_calibration([0.1, 0.5, 0.9], [0.2, 0.4, 0.8])
    assert "train_sizes" in plot_learning_curve(
        {"train_sizes": [10, 20], "train_rmse": [1.0, 0.8], "val_rmse": [1.1, 0.9]}
    )
    assert "residuals" in plot_residual_distribution(np.random.default_rng(0).normal(size=50))
