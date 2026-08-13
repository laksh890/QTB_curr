"""Integration tests for tree forecasting with simulation / regimes / volatility features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.forecasting.base.registry import get_registry
from iqrp.app.forecasting.tree_models import (
    TreeSettings,
    TreeTrainer,
    create_tree_model,
    ensure_tree_models_loaded,
)
from iqrp.app.forecasting.tree_models.base.processes import feature_names, simulate_nonlinear_returns
from iqrp.app.forecasting.tree_models.catboost.trainer import CatBoostTrainer
from iqrp.app.forecasting.tree_models.lightgbm.trainer import LightGBMTrainer
from iqrp.app.forecasting.tree_models.xgboost.trainer import XGBoostTrainer


@pytest.mark.integration
def test_end_to_end_tree_pipeline(tmp_path: Path) -> None:
    ensure_tree_models_loaded()
    assert "xgboost" in get_registry().list_names()
    frame = simulate_nonlinear_returns(260, n_features=6, rng=np.random.default_rng(11))
    cols = feature_names(6) + ["vol_forecast"]
    settings = TreeSettings.from_hydra(
        overrides=[
            "hyperparameters.n_estimators=40",
            "hyperparameters.max_depth=3",
            "feature_selection.enabled=true",
            "feature_selection.method=mutual_info",
            "feature_selection.max_features=5",
            "optimization.method=random",
            "optimization.n_trials=3",
            "regime.mode=feature",
            "visualization.enabled=false",
        ]
    )

    trainer = TreeTrainer(settings)
    model, result = trainer.fit("xgboost", frame, feature_columns=cols)
    assert result.metrics["rmse"] < 2.0
    # feature recovery: nonlinear f0/f1 should rank high often
    imp = model.feature_importance(kind="gain")
    assert max(imp.values()) > 0
    # regime adaptation
    fc = model.forecast(frame, horizon=5)
    assert fc.path().size == 5
    path = tmp_path / "xgb.json"
    model.save(path)
    loaded = type(model).load(path)
    assert loaded.evaluate(frame).metrics["n"] > 0


@pytest.mark.integration
def test_backend_trainers_and_generalization() -> None:
    frame = simulate_nonlinear_returns(180, n_features=4, rng=np.random.default_rng(12))
    cols = feature_names(4)
    settings = TreeSettings.from_mapping({"hyperparameters": {"n_estimators": 30, "max_depth": 3}})
    xgb = XGBoostTrainer(settings).fit(frame, feature_columns=cols)
    lgb = LightGBMTrainer(settings).fit(frame, feature_columns=cols)
    cat = CatBoostTrainer(settings).fit(frame, feature_columns=cols)
    # holdout generalization
    train, test = frame[:140], frame[140:]
    m = create_tree_model("hist_gradient_boosting", settings=settings)
    m.fit(train, feature_columns=cols)
    rmse = m.evaluate(test).metrics["rmse"]
    assert rmse < 3.0
    assert xgb.is_fitted and lgb.is_fitted and cat.is_fitted
