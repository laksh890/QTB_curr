"""Integration tests for neural forecasting with simulation / regimes / multi-horizon."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.forecasting.base.registry import get_registry
from iqrp.app.forecasting.neural import (
    NeuralOrchestrator,
    NeuralSettings,
    create_neural_model,
    ensure_neural_models_loaded,
)
from iqrp.app.forecasting.neural.base.processes import feature_names, simulate_nonlinear_returns


@pytest.mark.integration
def test_end_to_end_neural_pipeline(tmp_path: Path) -> None:
    ensure_neural_models_loaded()
    assert "lstm" in get_registry().list_names()
    frame = simulate_nonlinear_returns(200, n_features=5, rng=np.random.default_rng(11))
    cols = feature_names(5) + ["vol_forecast", "stat_forecast", "tree_forecast"]
    settings = NeuralSettings.from_hydra(
        overrides=[
            "architecture.lookback=16",
            "architecture.horizon=4",
            "architecture.hidden_size=24",
            "architecture.num_layers=1",
            "train.epochs=3",
            "train.batch_size=32",
            "train.device=cpu",
            "scheduler.name=cosine",
            "regime.mode=feature",
            "visualization.enabled=false",
            "optimization.method=none",
        ]
    )
    trainer = NeuralOrchestrator(settings)
    model, result = trainer.fit("lstm", frame, feature_columns=cols)
    assert result.metrics["rmse"] < 5.0
    fc = model.forecast(frame, horizon=4)
    assert fc.path().size == 4
    path = tmp_path / "lstm.json"
    model.save(path)
    loaded = type(model).load(path)
    assert loaded.evaluate(frame).metrics["n"] > 0


@pytest.mark.integration
def test_synthetic_forecast_recovery_and_generalization() -> None:
    frame = simulate_nonlinear_returns(220, n_features=4, noise=0.05, rng=np.random.default_rng(12))
    cols = feature_names(4)
    settings = NeuralSettings.from_mapping(
        {
            "architecture": {"lookback": 16, "horizon": 3, "hidden_size": 32, "num_layers": 1, "dropout": 0.0},
            "train": {"epochs": 8, "batch_size": 32, "device": "cpu", "learning_rate": 1e-3},
            "scheduler": {"name": "none"},
            "regime": {"enabled": False},
            "visualization": {"enabled": False},
        }
    )
    train, test = frame[:160], frame[160:]
    model = create_neural_model("mlp", settings=settings)
    model.fit(train, feature_columns=cols)
    rmse = model.evaluate(test).metrics["rmse"]
    assert rmse < 3.0
    # multi-architecture stress
    for name in ("tcn", "nbeats", "nhits", "seq2seq"):
        m = create_neural_model(name, settings=settings)
        m.fit(train, feature_columns=cols)
        assert m.forecast(test, horizon=3).path().size == 3


@pytest.mark.integration
def test_probabilistic_deepar_calibration() -> None:
    frame = simulate_nonlinear_returns(180, n_features=4, rng=np.random.default_rng(13))
    cols = feature_names(4)
    settings = NeuralSettings.from_mapping(
        {
            "task": {"type": "distribution"},
            "architecture": {"lookback": 12, "horizon": 3, "hidden_size": 24, "num_layers": 1},
            "train": {"epochs": 4, "batch_size": 32, "device": "cpu", "loss": "gaussian_nll"},
            "probabilistic": {"enabled": True, "distribution": "gaussian", "mc_dropout": True, "n_samples": 5},
            "scheduler": {"name": "none"},
            "regime": {"enabled": False},
            "visualization": {"enabled": False},
        }
    )
    model = create_neural_model("deepar", settings=settings)
    model.fit(frame, feature_columns=cols)
    fc = model.forecast(frame, horizon=3)
    assert "uncertainty" in fc.metadata or fc.intervals is not None
    intervals = model.forecast_interval(frame, horizon=3)
    assert len(intervals) == 3
