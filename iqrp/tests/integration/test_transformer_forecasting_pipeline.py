"""Integration tests for transformer forecasting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.forecasting.base.registry import get_registry
from iqrp.app.forecasting.transformers import (
    TransformerOrchestrator,
    TransformerSettings,
    create_transformer_model,
    ensure_transformer_models_loaded,
)
from iqrp.app.forecasting.transformers.base.processes import (
    feature_names,
    simulate_long_range_series,
)


@pytest.mark.integration
def test_end_to_end_transformer_pipeline(tmp_path: Path) -> None:
    ensure_transformer_models_loaded()
    assert "patchtst" in get_registry().list_names()
    frame = simulate_long_range_series(220, n_features=5, rng=np.random.default_rng(11))
    cols = feature_names(5) + ["vol_forecast", "neural_forecast", "tree_forecast", "stat_forecast"]
    settings = TransformerSettings.from_hydra(
        overrides=[
            "architecture.lookback=24",
            "architecture.horizon=6",
            "architecture.d_model=32",
            "architecture.n_heads=4",
            "architecture.num_layers=1",
            "architecture.patch_len=6",
            "architecture.stride=3",
            "train.epochs=3",
            "train.batch_size=16",
            "train.device=cpu",
            "scheduler.name=cosine",
            "regime.mode=embedding",
            "visualization.enabled=false",
        ]
    )
    trainer = TransformerOrchestrator(settings)
    model, result = trainer.fit("patchtst", frame, feature_columns=cols)
    assert result.metrics["rmse"] < 5.0
    fc = model.forecast(frame, horizon=6)
    assert fc.path().size == 6
    path = tmp_path / "pt.json"
    model.save(path)
    loaded = type(model).load(path)
    assert loaded.evaluate(frame).metrics["n"] > 0


@pytest.mark.integration
def test_long_horizon_generalization_and_multi_arch() -> None:
    frame = simulate_long_range_series(240, n_features=4, noise=0.05, rng=np.random.default_rng(12))
    cols = feature_names(4)
    settings = TransformerSettings.from_mapping(
        {
            "architecture": {
                "lookback": 32,
                "horizon": 8,
                "d_model": 32,
                "n_heads": 4,
                "num_layers": 1,
                "ffn_dim": 64,
                "dropout": 0.0,
                "patch_len": 8,
                "stride": 4,
                "moving_avg": 7,
            },
            "train": {"epochs": 6, "batch_size": 16, "device": "cpu", "learning_rate": 1e-3},
            "scheduler": {"name": "none"},
            "regime": {"enabled": False},
            "visualization": {"enabled": False},
        }
    )
    train, test = frame[:180], frame[180:]
    model = create_transformer_model("tide", settings=settings)
    model.fit(train, feature_columns=cols)
    rmse = model.evaluate(test).metrics["rmse"]
    assert rmse < 3.5
    for name in ("tft", "autoformer", "timesnet", "itransformer"):
        m = create_transformer_model(name, settings=settings)
        m.fit(train, feature_columns=cols)
        assert m.forecast(test, horizon=8).path().size == 8


@pytest.mark.integration
def test_quantile_probabilistic_and_regime() -> None:
    frame = simulate_long_range_series(180, n_features=4, rng=np.random.default_rng(13))
    cols = feature_names(4)
    settings = TransformerSettings.from_mapping(
        {
            "task": {"type": "quantile", "quantile_alphas": [0.1, 0.5, 0.9]},
            "architecture": {
                "lookback": 16,
                "horizon": 4,
                "d_model": 32,
                "n_heads": 4,
                "num_layers": 1,
            },
            "train": {"epochs": 3, "batch_size": 16, "device": "cpu", "loss": "quantile"},
            "probabilistic": {"enabled": True, "distribution": "quantile"},
            "regime": {"enabled": True, "mode": "embedding"},
            "scheduler": {"name": "none"},
            "visualization": {"enabled": False},
        }
    )
    model = create_transformer_model("tft", settings=settings)
    model.fit(frame, feature_columns=cols, regime_column="regime")
    intervals = model.forecast_interval(frame, horizon=4)
    assert len(intervals) == 4
