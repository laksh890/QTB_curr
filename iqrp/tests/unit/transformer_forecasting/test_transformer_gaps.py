"""Gap-closing tests for transformer forecasting coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from iqrp.app.forecasting.neural.base.losses import get_loss
from iqrp.app.forecasting.transformers import (
    TransformerSettings,
    create_transformer_model,
    ensure_transformer_models_loaded,
)
from iqrp.app.forecasting.transformers.attention import build_attention
from iqrp.app.forecasting.transformers.base.embeddings import (
    AssetEmbedding,
    CalendarEmbedding,
    CategoricalEmbedding,
    RegimeEmbedding,
    SectorEmbedding,
    TimeEmbedding,
    TransformerInputEmbedding,
)
from iqrp.app.forecasting.transformers.base.heads import forecast_head, reshape_forecast
from iqrp.app.forecasting.transformers.base.masking import apply_mask_to_scores, causal_mask
from iqrp.app.forecasting.transformers.base.processes import (
    feature_names,
    simulate_long_range_series,
)
from iqrp.app.forecasting.transformers.base.trainer import TransformerTrainer, _compute_loss
from iqrp.app.forecasting.transformers.explainability.attribution import explain_transformer
from iqrp.app.forecasting.transformers.mixture_of_experts import ExpertFFN, MoERouter
from iqrp.app.forecasting.transformers.probabilistic import (
    gaussian_head_quantiles,
    student_t_head_quantiles,
)
from iqrp.app.forecasting.transformers.visualization.plots import (
    plot_attention_map,
    plot_forecast,
    plot_training_curves,
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
            "chunk_size": 8,
        },
        "train": {
            "epochs": 1,
            "batch_size": 16,
            "device": "cpu",
            "early_stopping_patience": 20,
            "seed": 0,
        },
        "scheduler": {"name": "none"},
        "regime": {"enabled": False},
        "visualization": {"enabled": False},
        "online": {"mode": "refit", "refresh_every": 1, "finetune_epochs": 1, "window": 40},
    }
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return TransformerSettings.from_mapping(base)


@pytest.fixture
def frame():
    return simulate_long_range_series(100, n_features=3, n_assets=2, rng=np.random.default_rng(9))


@pytest.mark.unit
def test_embeddings_all() -> None:
    te = TimeEmbedding(16)
    assert te(torch.randn(2, 5, 3)).shape[-1] == 16
    assert AssetEmbedding(4, 8)(torch.tensor([0, 1])).shape == (2, 8)
    assert RegimeEmbedding(3, 8)(torch.tensor([0, 2])).shape == (2, 8)
    assert SectorEmbedding(5, 8)(torch.tensor([1, 2])).shape == (2, 8)
    assert CalendarEmbedding(8)(torch.tensor([1, 2]), torch.tensor([0, 11])).shape == (2, 8)
    assert CategoricalEmbedding(6, 8)(torch.tensor([0, 3])).shape == (2, 8)
    emb = TransformerInputEmbedding(3, 16, n_assets=2, use_regime=True)
    x = torch.randn(2, 5, 3)
    out = emb(x, regime_ids=torch.tensor([0, 1]), asset_ids=torch.tensor([0, 1]))
    assert out.shape == (2, 5, 16)


@pytest.mark.unit
def test_heads_classification_distribution() -> None:
    h = forecast_head(16, 4, task="classification", n_classes=3)
    out = h(torch.randn(2, 16))
    r = reshape_forecast(out, 2, 4, task="classification", n_classes=3)
    assert r.shape == (2, 4, 3)
    hd = forecast_head(16, 4, task="distribution")
    rd = reshape_forecast(hd(torch.randn(2, 16)), 2, 4, task="distribution")
    assert rd.shape[-1] == 2
    hq = forecast_head(16, 4, task="quantile", n_quantiles=3)
    assert (
        reshape_forecast(hq(torch.randn(2, 16)), 2, 4, task="quantile", n_quantiles=3).shape[-1]
        == 3
    )
    hm = forecast_head(16, 4, task="mixture", n_mixtures=2)
    assert hm(torch.randn(2, 16)).shape[-1] == 4 * 2 * 3


@pytest.mark.unit
def test_online_refit_and_no_torch(frame) -> None:
    cols = feature_names(3)
    m = create_transformer_model("tide", settings=_fast())
    m.fit(frame, feature_columns=cols)
    m.partial_fit(frame, feature_columns=cols)
    with patch(
        "iqrp.app.forecasting.transformers.base.transformer_model.has_torch", return_value=False
    ):
        with pytest.raises(Exception):
            create_transformer_model("tide", settings=_fast()).fit(frame, feature_columns=cols)


@pytest.mark.unit
def test_chunked_predict_and_loss_paths(frame) -> None:
    cols = feature_names(3)
    s = _fast(
        architecture={
            "lookback": 12,
            "horizon": 3,
            "d_model": 32,
            "n_heads": 4,
            "num_layers": 1,
            "chunk_size": 4,
        }
    )
    m = create_transformer_model("informer", settings=s)
    m.fit(frame, feature_columns=cols)
    # force chunked path
    trainer = TransformerTrainer(s)
    trainer.device = m._device
    X = np.random.randn(3, 20, 3).astype(np.float32)
    out = trainer.predict(m._module, X)
    assert out.shape[0] == 3
    loss_fn = get_loss("mse")
    _compute_loss(loss_fn, torch.randn(4, 3), torch.randn(4, 3), task="regression")
    _compute_loss(loss_fn, torch.randn(4, 3, 1), torch.randn(4, 3), task="regression")
    _compute_loss(get_loss("bce"), torch.randn(4, 3), torch.rand(4, 3), task="binary")
    _compute_loss(
        get_loss("cross_entropy"),
        torch.randn(4, 2, 3),
        torch.randint(0, 3, (8,)),
        task="classification",
    )


@pytest.mark.unit
def test_proba_unsupported_and_export(tmp_path: Path, frame) -> None:
    cols = feature_names(3)
    m = create_transformer_model("tide", settings=_fast())
    m.fit(frame, feature_columns=cols)
    from iqrp.app.forecasting.base.metadata import ForecastModelMeta

    object.__setattr__(
        m,
        "meta",
        ForecastModelMeta(
            name="tide",
            version="1.0.0",
            description="x",
            algorithm_family="transformer",
            task="regression",
            default_horizon=3,
            supports_online=True,
            supports_proba=False,
            supports_intervals=True,
            supports_quantiles=True,
        ),
    )
    with pytest.raises(Exception):
        m.predict_proba(frame)
    with patch("torch.onnx.export", side_effect=RuntimeError("no")):
        p = m.export_onnx(tmp_path / "a.onnx")
        assert p.exists()


@pytest.mark.unit
def test_attention_cross_asset_factory_and_mask_apply() -> None:
    a = build_attention("cross_asset", 16, 4)
    x = torch.randn(2, 3, 5, 16)
    assert a(x).shape == x.shape
    scores = torch.randn(2, 4, 5, 5)
    m = causal_mask(5)
    out = apply_mask_to_scores(scores, m)
    assert out.shape == scores.shape
    out2 = apply_mask_to_scores(scores, m.unsqueeze(0))
    assert out2.shape == scores.shape


@pytest.mark.unit
def test_prob_fallback_and_moe_exports() -> None:
    assert gaussian_head_quantiles(np.random.randn(3, 4)).shape[-1] == 3
    assert student_t_head_quantiles(np.random.randn(3, 4)).shape[-1] == 3
    assert ExpertFFN(8, 16)(torch.randn(2, 8)).shape == (2, 8)
    w, top = MoERouter(8, 4, top_k=2)(torch.randn(2, 5, 8))
    assert w.shape[-1] == 4
    assert ensure_transformer_models_loaded(["iqrp.does.not.exist"]) == []


@pytest.mark.unit
def test_plots_no_pyplot() -> None:
    with patch("iqrp.app.forecasting.transformers.visualization.plots._pyplot", return_value=None):
        assert "attention" in plot_attention_map(np.ones((4, 4)))
        assert "y_pred" in plot_forecast(np.arange(3))
        assert "train_loss" in plot_training_curves({"train_loss": [1.0]})


@pytest.mark.unit
def test_settings_default_missing_file() -> None:
    with patch("iqrp.app.forecasting.transformers.config._default_config_path") as p:
        p.return_value = Path("/tmp/no_tx_config.yaml")
        s = TransformerSettings.default()
        assert s.train.epochs > 0
    from omegaconf import OmegaConf

    TransformerSettings.from_mapping(OmegaConf.create({"train": {"epochs": 2}}))


@pytest.mark.unit
def test_flash_chunked_fallback() -> None:
    from iqrp.app.forecasting.transformers.attention.flash_attention import FlashAttention

    fa = FlashAttention(32, 4, chunk_size=4)
    q = torch.randn(1, 4, 8, 8)
    # call internal chunked
    out = fa._chunked(q, q, q)
    assert out.shape == q.shape


@pytest.mark.unit
def test_explain_no_torch(frame) -> None:
    cols = feature_names(3)
    m = create_transformer_model("tide", settings=_fast())
    m.fit(frame, feature_columns=cols)
    X = m._last_window(frame, cols)
    with patch(
        "iqrp.app.forecasting.transformers.explainability.attribution.has_torch", return_value=False
    ):
        assert explain_transformer(m._module, X, method="ig").shape == X.shape
