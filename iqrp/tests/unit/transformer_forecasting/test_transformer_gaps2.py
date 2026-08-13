"""Additional coverage gaps for transformers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from iqrp.app.forecasting.transformers import TransformerSettings, create_transformer_model
from iqrp.app.forecasting.transformers.attention.temporal_attention import HierarchicalAttention
from iqrp.app.forecasting.transformers.base.masking import (
    causal_mask,
    local_attention_mask,
    padding_mask,
    regime_mask,
)
from iqrp.app.forecasting.transformers.base.processes import feature_names, simulate_long_range_series
from iqrp.app.forecasting.transformers.base.trainer import TransformerTrainer
from iqrp.app.forecasting.transformers.diagnostics.report import run_transformer_diagnostics
from iqrp.app.forecasting.transformers.probabilistic import mixture_density_params, mixture_mean
from iqrp.app.forecasting.transformers.visualization.plots import (
    plot_embedding_projection,
    plot_residual_distribution,
    plot_calibration_curve,
)


def _fast(**extra):
    base = {
        "architecture": {
            "lookback": 12,
            "horizon": 3,
            "d_model": 32,
            "n_heads": 4,
            "num_layers": 1,
            "ffn_dim": 64,
            "dropout": 0.0,
            "patch_len": 4,
            "stride": 2,
            "moving_avg": 5,
        },
        "train": {"epochs": 1, "batch_size": 16, "device": "cpu", "early_stopping_patience": 20},
        "scheduler": {"name": "none"},
        "regime": {"enabled": False},
        "visualization": {"enabled": False},
        "online": {"mode": "warm_start", "refresh_every": 1, "window": 50, "finetune_epochs": 1},
    }
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return TransformerSettings.from_mapping(base)


@pytest.fixture
def frame():
    return simulate_long_range_series(90, n_features=3, rng=np.random.default_rng(5))


@pytest.mark.unit
def test_masking_no_torch() -> None:
    with patch("iqrp.app.forecasting.transformers.base.masking.has_torch", return_value=False):
        assert causal_mask(4) is None
        assert padding_mask(torch.tensor([1, 2]), 3) is None
        assert local_attention_mask(4, 2) is None
        assert regime_mask(torch.tensor([0, 1]), allow_cross=False) is None


@pytest.mark.unit
def test_hierarchical_padding() -> None:
    h = HierarchicalAttention(16, 4, pool=3)
    x = torch.randn(2, 7, 16)  # not divisible by 3
    assert h(x).shape == x.shape


@pytest.mark.unit
def test_diagnostics_edges_and_mixture_2d() -> None:
    class Dummy:
        _residuals = np.array([])
        _history = MagicMock(to_dict=lambda: {})
        architecture_name = "x"
        _module = None
        _last_attn = None
        _last_embeddings = None
        _X_seq = None

    assert run_transformer_diagnostics(Dummy()).residual_mean == 0.0

    class Dummy2:
        _residuals = np.array([1.0, -1.0])
        _history = MagicMock(to_dict=lambda: {})
        architecture_name = "x"
        _module = torch.nn.Linear(2, 1)
        _last_attn = np.ones((1, 4, 4)) / 4
        _last_embeddings = np.ones((1, 2, 2))
        _X_seq = np.zeros((5, 2, 2))

    d = run_transformer_diagnostics(Dummy2()).to_dict()
    assert d["attention_entropy"] >= 0

    pred = np.random.randn(4, 9)
    md = mixture_density_params(pred, n_mixtures=3)
    assert md["pi"].ndim == 3
    assert mixture_mean(pred, 3).shape[0] == 4


@pytest.mark.unit
def test_warm_start_partial_and_no_features(frame) -> None:
    cols = feature_names(3)
    m = create_transformer_model("informer", settings=_fast())
    m.fit(frame[:70], feature_columns=cols)
    m.partial_fit(frame, feature_columns=cols)
    with pytest.raises(Exception):
        create_transformer_model("tide", settings=_fast()).fit(
            frame.select(["open_time", "target"]), feature_columns=[]
        )
    with pytest.raises(Exception):
        create_transformer_model("tide", settings=_fast())._resolve_target(frame.select(cols), None)


@pytest.mark.unit
def test_trainer_no_torch_and_plateau(frame) -> None:
    s = _fast()
    trainer = TransformerTrainer(s)
    with patch("iqrp.app.forecasting.transformers.base.trainer.has_torch", return_value=False):
        with pytest.raises(RuntimeError):
            trainer.fit(torch.nn.Linear(2, 1), np.ones((4, 2)), np.ones(4))


@pytest.mark.unit
def test_forecast_pad_and_dict_settings(frame) -> None:
    cols = feature_names(3)
    m = create_transformer_model(
        "patchtst",
        settings={
            "architecture": {"lookback": 12, "horizon": 2, "d_model": 32, "n_heads": 4, "num_layers": 1, "patch_len": 4, "stride": 2},
            "train": {"epochs": 1, "device": "cpu"},
            "scheduler": {"name": "none"},
            "regime": {"enabled": False},
            "visualization": {"enabled": False},
        },
    )
    m.fit(frame, feature_columns=cols)
    fc = m.forecast(frame, horizon=7)
    assert fc.path().size == 7
    # classification 3d predict_proba
    cls = simulate_long_range_series(80, n_features=3, classification=True, rng=np.random.default_rng(8))
    m2 = create_transformer_model(
        "tft",
        settings=_fast(task={"type": "classification", "n_classes": 2}, train={"loss": "cross_entropy", "epochs": 1, "device": "cpu"}),
    )
    m2.fit(cls, feature_columns=cols)
    assert m2.predict_proba(cls).shape[1] == 2


@pytest.mark.unit
def test_viz_embedding_1d_and_export_no_torch(tmp_path: Path, frame) -> None:
    plot_embedding_projection(np.random.randn(10, 1))
    plot_residual_distribution(np.ones(5))
    plot_calibration_curve([0.2, 0.8], [0.25, 0.75])
    cols = feature_names(3)
    m = create_transformer_model("tide", settings=_fast())
    m.fit(frame, feature_columns=cols)
    with patch("iqrp.app.forecasting.transformers.base.transformer_model.has_torch", return_value=False):
        with pytest.raises(Exception):
            m.export_onnx(tmp_path / "x.onnx")


@pytest.mark.unit
def test_gradient_checkpoint_flag(frame) -> None:
    cols = feature_names(3)
    s = _fast(train={"epochs": 1, "device": "cpu", "gradient_checkpointing": True, "ema_decay": 0.5})
    create_transformer_model("timesnet", settings=s).fit(frame, feature_columns=cols)
