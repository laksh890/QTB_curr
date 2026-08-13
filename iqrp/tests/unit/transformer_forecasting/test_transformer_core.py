"""Core unit tests for Institutional Transformer Forecasting Platform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from iqrp.app.forecasting.transformers import (
    TransformerOrchestrator,
    TransformerSettings,
    create_transformer_model,
    list_transformer_models,
)
from iqrp.app.forecasting.transformers.attention import build_attention
from iqrp.app.forecasting.transformers.base.masking import (
    causal_mask,
    combine_masks,
    local_attention_mask,
    padding_mask,
    regime_mask,
)
from iqrp.app.forecasting.transformers.base.positional_encoding import build_positional
from iqrp.app.forecasting.transformers.base.processes import feature_names, simulate_long_range_series
from iqrp.app.forecasting.transformers.explainability.attribution import explain_transformer
from iqrp.app.forecasting.transformers.mixture_of_experts import MoEGating
from iqrp.app.forecasting.transformers.probabilistic import (
    gaussian_head_quantiles,
    mixture_density_params,
    mixture_mean,
    student_t_head_quantiles,
)
from iqrp.app.forecasting.transformers.visualization.plots import (
    plot_attention_map,
    plot_calibration_curve,
    plot_embedding_projection,
    plot_forecast,
    plot_residual_distribution,
    plot_training_curves,
)


def _fast(**extra):
    base = {
        "architecture": {
            "lookback": 16,
            "horizon": 4,
            "d_model": 32,
            "n_heads": 4,
            "num_layers": 1,
            "ffn_dim": 64,
            "dropout": 0.0,
            "patch_len": 4,
            "stride": 2,
            "moving_avg": 5,
        },
        "train": {
            "epochs": 2,
            "batch_size": 16,
            "device": "cpu",
            "early_stopping_patience": 20,
            "seed": 0,
            "optimizer": "adamw",
        },
        "scheduler": {"name": "none"},
        "regime": {"enabled": False},
        "visualization": {"enabled": False},
        "online": {"mode": "finetune", "finetune_epochs": 1, "window": 80, "refresh_every": 1000},
    }
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return TransformerSettings.from_mapping(base)


@pytest.fixture
def frame():
    return simulate_long_range_series(140, n_features=4, rng=np.random.default_rng(1))


@pytest.mark.unit
def test_registry_lists_all_models() -> None:
    names = set(list_transformer_models())
    assert names >= {
        "tft",
        "informer",
        "autoformer",
        "fedformer",
        "patchtst",
        "crossformer",
        "timesnet",
        "itransformer",
        "timemixer",
        "tide",
        "moe_transformer",
    }


@pytest.mark.unit
def test_settings_hydra() -> None:
    s = TransformerSettings.default()
    assert s.architecture.d_model > 0
    s2 = TransformerSettings.from_mapping({"task": {"type": "quantile"}, "train": {"loss": "quantile"}})
    assert s2.task.type == "quantile"
    s3 = TransformerSettings.from_hydra(overrides=["forecast.default_horizon=12"])
    assert s3.forecast.default_horizon == 12
    with pytest.raises(Exception):
        TransformerSettings.from_mapping({"task": {"type": "not_a_task"}})


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        "tft",
        "informer",
        "autoformer",
        "fedformer",
        "patchtst",
        "crossformer",
        "timesnet",
        "itransformer",
        "timemixer",
        "tide",
        "moe_transformer",
    ],
)
def test_all_models_api(name: str, frame) -> None:
    model = create_transformer_model(name, settings=_fast())
    cols = feature_names(4)
    model.fit(frame, feature_columns=cols)
    pred = model.predict(frame)
    assert pred.shape[0] == frame.height
    fc = model.forecast(frame, horizon=4)
    assert fc.path().shape == (4,)
    assert len(model.forecast_interval(frame, horizon=3)) == 3
    report = model.evaluate(frame)
    assert "rmse" in report.metrics
    expl = model.explain(frame, method="attention_rollout")
    assert expl.importances
    diag = model.diagnostics()
    assert "residual_std" in diag
    attn = model.attention(frame)
    assert attn.size >= 1
    emb = model.embeddings(frame)
    assert emb.size >= 1


@pytest.mark.unit
def test_partial_fit_save_load_export(tmp_path: Path, frame) -> None:
    cols = feature_names(4)
    model = create_transformer_model("patchtst", settings=_fast())
    model.fit(frame[:100], feature_columns=cols)
    model.partial_fit(frame, feature_columns=cols)
    path = tmp_path / "pt.json"
    model.save(path)
    loaded = type(model).load(path)
    assert loaded.evaluate(frame).metrics["n"] > 0
    out = model.export_onnx(tmp_path / "m.onnx")
    assert out.exists()


@pytest.mark.unit
def test_attention_builders_and_masks() -> None:
    x = torch.randn(2, 10, 32)
    for name in ("full", "flash", "sparse", "linear", "performer", "temporal", "hierarchical"):
        attn = build_attention(name, 32, 4, 0.0)
        if name in {"temporal", "hierarchical"}:
            y = attn(x)
        else:
            y = attn(x, x, x)
        assert y.shape == x.shape
    cm = causal_mask(8)
    assert cm.shape == (8, 8)
    pm = padding_mask(torch.tensor([5, 7]), 8)
    assert pm.shape == (2, 8)
    lm = local_attention_mask(8, 2)
    assert lm.shape == (8, 8)
    rm = regime_mask(torch.tensor([[0, 0, 1, 1]]), allow_cross=False)
    assert rm is not None
    assert combine_masks(cm, None) is not None
    for pos in ("sinusoidal", "learned", "rotary"):
        p = build_positional(pos, 32, max_len=64)
        if pos == "rotary":
            q = torch.randn(2, 4, 10, 32)
            qq, kk = p(q, q)
            assert qq.shape == q.shape
        else:
            assert p(x).shape == x.shape


@pytest.mark.unit
def test_moe_and_probabilistic() -> None:
    moe = MoEGating(32, n_experts=3, ffn_dim=64)
    h = torch.randn(2, 8, 32)
    assert moe(h).shape == h.shape
    pred = np.random.randn(4, 5, 9)
    md = mixture_density_params(pred, n_mixtures=3)
    assert "pi" in md
    assert mixture_mean(pred, 3).shape[:2] == (4, 5)
    g = gaussian_head_quantiles(np.random.randn(4, 5, 2))
    assert g.shape[-1] == 3
    st = student_t_head_quantiles(np.random.randn(4, 5, 2))
    assert st.shape == g.shape


@pytest.mark.unit
def test_classification_and_orchestrator(frame) -> None:
    cls = simulate_long_range_series(120, n_features=4, classification=True, rng=np.random.default_rng(3))
    cols = feature_names(4)
    model = create_transformer_model(
        "tft",
        settings=_fast(task={"type": "binary"}, train={"loss": "bce", "epochs": 2, "device": "cpu"}),
    )
    model.fit(cls, feature_columns=cols)
    proba = model.predict_proba(cls)
    assert proba.shape[1] == 2
    orch = TransformerOrchestrator(_fast(visualization={"enabled": True}))
    m2, result = orch.fit("tide", frame, feature_columns=cols)
    assert result.metrics
    assert m2.is_fitted
    assert result.to_dict()["model_name"] == "tide"


@pytest.mark.unit
def test_explain_methods_and_viz(frame) -> None:
    cols = feature_names(4)
    model = create_transformer_model("informer", settings=_fast())
    model.fit(frame, feature_columns=cols)
    X = model._last_window(frame, cols)
    for method in ("attention_rollout", "ig", "saliency", "token"):
        attr = explain_transformer(model._module, X, method=method, device=model._device)
        assert np.asarray(attr).size > 0
    plot_attention_map(np.random.rand(8, 8))
    plot_forecast(np.arange(5), y_true=np.arange(5), bands=(np.arange(5) - 1, np.arange(5) + 1))
    plot_embedding_projection(np.random.randn(20, 8))
    plot_residual_distribution(np.random.randn(40))
    plot_calibration_curve([0.1, 0.5, 0.9], [0.12, 0.48, 0.88])
    plot_training_curves({"train_loss": [1.0, 0.5], "val_loss": [1.1, 0.6]})


@pytest.mark.unit
def test_regime_feature_and_curriculum(frame) -> None:
    cols = feature_names(4)
    s = _fast(
        regime={"enabled": True, "mode": "feature", "column": "regime"},
        train={"epochs": 2, "batch_size": 16, "device": "cpu", "curriculum": True, "ema_decay": 0.9},
        scheduler={"name": "warmup_cosine", "warmup_epochs": 1},
    )
    model = create_transformer_model("moe_transformer", settings=s)
    model.fit(frame, feature_columns=cols, regime_column="regime")
    assert model.is_fitted


@pytest.mark.unit
def test_cross_asset_attention() -> None:
    from iqrp.app.forecasting.transformers.attention.cross_asset_attention import CrossAssetAttention

    ca = CrossAssetAttention(16, 4)
    x = torch.randn(2, 3, 8, 16)
    assert ca(x).shape == x.shape
