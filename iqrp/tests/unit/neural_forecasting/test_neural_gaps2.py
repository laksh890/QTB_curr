"""Additional coverage gaps for neural forecasting."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from iqrp.app.forecasting.neural import NeuralSettings, create_neural_model
from iqrp.app.forecasting.neural.base.data import train_val_split
from iqrp.app.forecasting.neural.base.losses import get_loss
from iqrp.app.forecasting.neural.base.processes import feature_names, simulate_nonlinear_returns
from iqrp.app.forecasting.neural.base.scheduler import WarmupCosineScheduler, build_optimizer
from iqrp.app.forecasting.neural.base.trainer import NeuralTrainer, _compute_loss
from iqrp.app.forecasting.neural.deepar.model import DeepARForecastModel
from iqrp.app.forecasting.neural.diagnostics.report import run_neural_diagnostics
from iqrp.app.forecasting.neural.explainability.attribution import explain_neural
from iqrp.app.forecasting.neural.nbeats.net import NBeatsNet
from iqrp.app.forecasting.neural.nhits.net import NHitsNet
from iqrp.app.forecasting.neural.optimization.distributed import wrap_ddp
from iqrp.app.forecasting.neural.probabilistic.distributions import epistemic_mc_dropout
from iqrp.app.forecasting.neural.probabilistic.quantiles import quantiles_from_prediction
from iqrp.app.forecasting.neural.trainer import NeuralOrchestrator
from iqrp.app.forecasting.neural.variants import _merge
from iqrp.app.forecasting.neural.visualization.plots import (
    plot_attention,
    plot_attribution,
    plot_forecast,
    plot_loss_curve,
    plot_residual_distribution,
    plot_training_curves,
)


def _fast(**extra):
    base = {
        "architecture": {"lookback": 8, "horizon": 2, "hidden_size": 8, "num_layers": 1, "n_blocks": 3, "dropout": 0.0},
        "train": {"epochs": 1, "batch_size": 16, "device": "cpu", "early_stopping_patience": 20, "seed": 0},
        "scheduler": {"name": "none"},
        "regime": {"enabled": False},
        "visualization": {"enabled": False},
    }
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return NeuralSettings.from_mapping(base)


@pytest.fixture
def frame():
    return simulate_nonlinear_returns(90, n_features=3, rng=np.random.default_rng(21))


@pytest.mark.unit
def test_no_torch_fit_raises(frame) -> None:
    m = create_neural_model("mlp", settings=_fast())
    with patch("iqrp.app.forecasting.neural.base.neural_model.has_torch", return_value=False):
        with pytest.raises(Exception):
            m.fit(frame, feature_columns=feature_names(3))


@pytest.mark.unit
def test_no_target_and_align_proba(frame) -> None:
    cols = feature_names(3)
    m = create_neural_model("mlp", settings=_fast())
    with pytest.raises(Exception):
        m._resolve_target(frame.select(cols), None)
    m.fit(frame, feature_columns=cols)
    # force 1d proba align
    out = m._align_proba(np.array([0.2, 0.8, 0.1]), frame.height)
    assert out.shape[1] == 2


@pytest.mark.unit
def test_classification_3d_proba(frame) -> None:
    cols = feature_names(3)
    cls = simulate_nonlinear_returns(90, n_features=3, classification=True, rng=np.random.default_rng(4))
    s = _fast(task={"type": "classification", "n_classes": 2}, train={"loss": "cross_entropy", "epochs": 2, "device": "cpu"})
    m = create_neural_model("lstm", settings=s)
    m.fit(cls, feature_columns=cols)
    proba = m.predict_proba(cls)
    assert proba.ndim == 2


@pytest.mark.unit
def test_export_onnx_errors(frame, tmp_path: Path) -> None:
    cols = feature_names(3)
    m = create_neural_model("mlp", settings=_fast())
    m.fit(frame, feature_columns=cols)
    with patch("iqrp.app.forecasting.neural.base.neural_model.has_torch", return_value=False):
        with pytest.raises(Exception):
            m.export_onnx(tmp_path / "a.onnx")
    with patch("torch.onnx.export", side_effect=TypeError("no dynamo")):
        # second call also fails -> falls to pt
        with patch("torch.onnx.export", side_effect=[TypeError("x"), RuntimeError("y")]):
            path = m.export_onnx(tmp_path / "b.onnx")
            assert path.exists()


@pytest.mark.unit
def test_trainer_no_torch_and_tuple_out() -> None:
    settings = _fast()
    trainer = NeuralTrainer(settings)
    with patch("iqrp.app.forecasting.neural.base.trainer.has_torch", return_value=False):
        with pytest.raises(RuntimeError):
            trainer.fit(nn.Linear(2, 1), np.ones((4, 2)), np.ones(4))
    class Tup(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 2), torch.zeros(1)
    pred = trainer.predict(Tup(), np.random.randn(4, 3, 2).astype(np.float32) if False else np.random.randn(4, 8))
    # wrong shape - use proper
    mod = Tup()
    X = np.random.randn(4, 8).astype(np.float32)
    # Linear expects (B,F) - Tup ignores
    out = trainer.predict(mod, X)
    assert out.ndim >= 1
    loss = _compute_loss(get_loss("mse"), torch.randn(4, 3), torch.randn(4, 5), task="regression")
    assert float(loss.item()) >= 0


@pytest.mark.unit
def test_deepar_dict_without_train_key(frame) -> None:
    m2 = DeepARForecastModel(
        settings={
            "architecture": {"lookback": 8, "horizon": 2, "hidden_size": 8, "num_layers": 1},
            "regime": {"enabled": False},
        }
    )
    assert m2._neural_settings.train.loss == "gaussian_nll"
    m2.fit(frame, feature_columns=feature_names(3))


@pytest.mark.unit
def test_diagnostics_empty_and_weights() -> None:
    class Dummy:
        _residuals = np.array([0.1, -0.2, 0.0])
        _history = MagicMock(to_dict=lambda: {})
        architecture_name = "x"
        _module = nn.Linear(2, 1)
        _X_seq = np.random.randn(10, 4, 2)
        _neural_settings = _fast()
        _device = "cpu"

    rep = run_neural_diagnostics(Dummy())
    assert "weight_stats" in rep.to_dict()

    class EmptyParams(nn.Module):
        def parameters(self, recurse=True):
            return iter([])

    class Dummy2:
        _residuals = np.array([1.0])
        _history = MagicMock(to_dict=lambda: {"grad_norms": []})
        architecture_name = "x"
        _module = EmptyParams()
        _X_seq = None

    assert run_neural_diagnostics(Dummy2()).weight_stats == {}

    # empty residual calibration path
    class Dummy3:
        _residuals = np.array([])
        _history = MagicMock(to_dict=lambda: {})
        architecture_name = "x"
        _module = None
        _X_seq = None

    assert run_neural_diagnostics(Dummy3()).residual_mean == 0.0


@pytest.mark.unit
def test_nbeats_nhits_dist_and_pools() -> None:
    nb = NBeatsNet(3, 8, 2, hidden_size=8, n_blocks=1, task="distribution", dist=True)
    assert nb(torch.randn(2, 8, 3)).shape[-1] == 2
    nh = NHitsNet(3, 8, 2, hidden_size=8, n_blocks=5, task="distribution", dist=True)
    assert nh(torch.randn(2, 8, 3)).shape[-1] == 2


@pytest.mark.unit
def test_quantiles_task_quantile_path() -> None:
    q = np.random.randn(2, 3, 3)
    out = quantiles_from_prediction(q, task="quantile", alphas=(0.1, 0.5, 0.9))
    assert out.shape == q.shape


@pytest.mark.unit
def test_epistemic_no_torch() -> None:
    with patch("iqrp.app.forecasting.neural.probabilistic.distributions.has_torch", return_value=False):
        mean, std = epistemic_mc_dropout(lambda x: x.mean(axis=1), np.ones((2, 3, 1)), n_samples=2)
        assert mean.shape[0] == 2


@pytest.mark.unit
def test_wrap_ddp_initialized() -> None:
    lin = nn.Linear(2, 1)
    with patch("torch.distributed.is_available", return_value=True), patch(
        "torch.distributed.is_initialized", return_value=True
    ), patch("torch.distributed.get_world_size", return_value=2), patch(
        "torch.nn.parallel.DistributedDataParallel", side_effect=lambda m, **k: m
    ):
        assert wrap_ddp(lin, enabled=True) is lin


@pytest.mark.unit
def test_variants_merge_branches() -> None:
    s, _ = _merge(None, {"architecture": {"num_layers": 3}})
    assert s.architecture.num_layers == 3
    s2, _ = _merge({"architecture": {"hidden_size": 32}}, {"architecture": {"num_layers": 2}})
    assert s2.architecture.hidden_size == 32
    s3, _ = _merge(NeuralSettings.default(), {"train": {"epochs": 2}})
    assert s3.train.epochs == 2
    s4, _ = _merge(object(), {"train": {"epochs": 1}})
    assert s4.train.epochs == 1


@pytest.mark.unit
def test_plots_without_pyplot() -> None:
    with patch("iqrp.app.forecasting.neural.visualization.plots._pyplot", return_value=None):
        assert "y_pred" in plot_forecast(np.arange(3), np.arange(3))
        assert "train_loss" in plot_training_curves({"train_loss": [1.0]})
        assert "attention" in plot_attention(np.ones((2, 3)))
        assert "scores" in plot_attribution(np.ones((2, 3, 4)))
        assert "residuals" in plot_residual_distribution(np.ones(5))
        assert "losses" in plot_loss_curve([1.0, 0.5])


@pytest.mark.unit
def test_train_val_split_tiny() -> None:
    X = np.ones((1, 2))
    y = np.ones(1)
    a, b, c, d = train_val_split(X, y, val_ratio=0.5)
    assert c.shape[0] == 0


@pytest.mark.unit
def test_unknown_loss_and_lion_warmup(frame) -> None:
    assert get_loss("unknown_xyz") is not None
    lin = nn.Linear(2, 1)
    opt = build_optimizer(lin.parameters(), name="lion", lr=1e-3, weight_decay=0.0)
    sch = WarmupCosineScheduler(opt, warmup_epochs=1, total_epochs=3)
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    opt.zero_grad()
    ((lin(x) - y) ** 2).mean().backward()
    opt.step()
    sch.step()
    # lion with no grad
    opt.zero_grad(set_to_none=True)
    opt.step()


@pytest.mark.unit
def test_orchestrator_result_to_dict(frame) -> None:
    orch = NeuralOrchestrator(_fast())
    _, res = orch.fit("mlp", frame, feature_columns=feature_names(3))
    d = res.to_dict()
    assert d["model_name"] == "mlp"


@pytest.mark.unit
def test_regime_separate_skips_small(frame) -> None:
    # make almost all one regime so other is tiny
    f = frame.with_columns(__import__("polars").lit(0).alias("regime"))
    s = _fast(regime={"enabled": True, "mode": "separate", "column": "regime"})
    m = create_neural_model("mlp", settings=s)
    m.fit(f, feature_columns=feature_names(3), regime_column="regime")
    assert m.is_fitted


@pytest.mark.unit
def test_shap_fallback(frame) -> None:
    cols = feature_names(3)
    m = create_neural_model("mlp", settings=_fast())
    m.fit(frame, feature_columns=cols)
    X = m._last_window(frame, cols)
    with patch.dict("sys.modules", {"shap": MagicMock(DeepExplainer=MagicMock(side_effect=RuntimeError("no")))}):
        attr = explain_neural(m._module, X, method="shap", device=m._device)
        assert np.asarray(attr).size > 0


@pytest.mark.unit
def test_forecast_pad_short_horizon(frame) -> None:
    cols = feature_names(3)
    m = create_neural_model("mlp", settings=_fast(architecture={"horizon": 2, "lookback": 8, "hidden_size": 8}))
    m.fit(frame, feature_columns=cols)
    fc = m.forecast(frame, horizon=6)
    assert fc.path().size == 6
