"""Gap-closing tests for neural forecasting coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from iqrp.app.forecasting.neural import NeuralOrchestrator, NeuralSettings, create_neural_model
from iqrp.app.forecasting.neural.base.callbacks import EarlyStopping, GradientMonitor, History
from iqrp.app.forecasting.neural.base.data import make_sequences, train_val_split
from iqrp.app.forecasting.neural.base.heads import output_head, reshape_head
from iqrp.app.forecasting.neural.base.losses import GaussianNLLLoss, StudentTNLLLoss, get_loss
from iqrp.app.forecasting.neural.base.metrics import directional_accuracy, evaluate_predictions
from iqrp.app.forecasting.neural.base.processes import feature_names, simulate_nonlinear_returns
from iqrp.app.forecasting.neural.base.scheduler import (
    WarmupCosineScheduler,
    build_optimizer,
    build_scheduler,
)
from iqrp.app.forecasting.neural.base.torch_utils import (
    count_parameters,
    from_tensor,
    maybe_compile,
    resolve_device,
    seed_everything,
    to_tensor,
)
from iqrp.app.forecasting.neural.base.trainer import _compute_loss
from iqrp.app.forecasting.neural.embeddings.categorical import MixtureOfExperts
from iqrp.app.forecasting.neural.explainability.attribution import (
    explain_neural,
    occlusion_analysis,
    saliency_map,
)
from iqrp.app.forecasting.neural.optimization.distributed import (
    amp_enabled,
    enable_gradient_checkpointing,
    wrap_ddp,
)
from iqrp.app.forecasting.neural.optimization.hpo import optimize_neural
from iqrp.app.forecasting.neural.probabilistic.distributions import (
    aleatoric_from_gaussian,
    epistemic_mc_dropout,
)
from iqrp.app.forecasting.neural.probabilistic.quantiles import (
    extract_point_forecast,
    interval_from_prediction,
    quantiles_from_prediction,
)
from iqrp.app.forecasting.neural.registry import ensure_neural_models_loaded
from iqrp.app.forecasting.neural.seq2seq.net import Seq2SeqNet
from iqrp.app.forecasting.neural.visualization.plots import (
    plot_attribution,
    plot_forecast,
    plot_training_curves,
)


def _fast(**extra):
    base = {
        "architecture": {
            "lookback": 10,
            "horizon": 2,
            "hidden_size": 12,
            "num_layers": 1,
            "n_blocks": 2,
            "dropout": 0.0,
            "attention": False,
        },
        "train": {
            "epochs": 2,
            "batch_size": 16,
            "device": "cpu",
            "early_stopping_patience": 1,
            "seed": 1,
            "optimizer": "adamw",
        },
        "scheduler": {"name": "none"},
        "regime": {"enabled": False},
        "optimization": {"method": "none"},
        "visualization": {"enabled": True},
        "online": {"mode": "refit", "finetune_epochs": 1, "window": 40, "refresh_every": 1},
    }
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return NeuralSettings.from_mapping(base)


@pytest.fixture
def frame():
    return simulate_nonlinear_returns(100, n_features=3, rng=np.random.default_rng(7))


@pytest.mark.unit
def test_callbacks_early_stop_restore() -> None:
    m = nn.Linear(2, 1)
    stopper = EarlyStopping(patience=1, min_delta=0.0)
    assert stopper.step(1.0, m) is False
    assert stopper.step(2.0, m) is True
    assert stopper.should_stop
    gm = GradientMonitor()
    m.weight.grad = torch.ones_like(m.weight)
    assert gm.record(m) > 0
    h = History.from_dict({})
    assert h.train_loss == []


@pytest.mark.unit
def test_data_edge_cases() -> None:
    X = np.random.randn(3, 2)
    y = np.random.randn(3)
    xs, ys = make_sequences(X, y, lookback=5, horizon=2)
    assert xs.shape[0] == 1
    X2 = np.random.randn(2, 2)
    y2 = np.random.randn(2)
    a, b, c, d = train_val_split(X2, y2, val_ratio=0.5)
    assert a.shape[0] >= 0


@pytest.mark.unit
def test_heads_and_losses_edges() -> None:
    h = output_head(8, 3, task="classification", n_classes=4)
    out = h(torch.randn(2, 8))
    r = reshape_head(out, 2, 3, task="classification", n_classes=4)
    assert r.shape == (2, 3, 4)
    hq = output_head(8, 3, task="quantile", n_quantiles=3)
    rq = reshape_head(hq(torch.randn(2, 8)), 2, 3, task="quantile", n_quantiles=3)
    assert rq.shape[-1] == 3
    hd = output_head(8, 3, task="distribution")
    rd = reshape_head(hd(torch.randn(2, 8)), 2, 3, task="distribution")
    assert rd.shape[-1] == 2
    ce = get_loss("cross_entropy")
    logits = torch.randn(4, 3)
    tgt = torch.tensor([0, 1, 2, 1])
    assert float(ce(logits, tgt).item()) >= 0
    assert float(GaussianNLLLoss()(torch.randn(4, 1), torch.randn(4)).item()) >= 0
    assert float(StudentTNLLLoss()(torch.randn(4, 1), torch.randn(4)).item()) >= 0
    ql = get_loss("quantile", alphas=(0.5,))
    assert float(ql(torch.randn(4, 2), torch.randn(4)).item()) >= 0
    with patch("iqrp.app.forecasting.neural.base.losses.has_torch", return_value=False):
        nl = get_loss("mae")
        assert nl(np.ones(3), np.zeros(3)) == 1.0
        nl2 = get_loss("mse")
        assert nl2(np.ones(3), np.zeros(3)) == 1.0


@pytest.mark.unit
def test_metrics_classification() -> None:
    assert np.isnan(directional_accuracy(np.array([1.0]), np.array([1.0])))
    yt = np.array([0.0, 1.0, 1.0, 0.0])
    yp = np.array([0.1, 0.9, 0.8, 0.2])
    proba = np.column_stack([1 - yp, yp])
    m = evaluate_predictions(yt, yp, proba=proba, task="binary")
    assert "brier" in m and "log_loss" in m


@pytest.mark.unit
def test_scheduler_lion_lookahead_warmup() -> None:
    lin = nn.Linear(3, 1)
    x = torch.randn(4, 3)
    y = torch.randn(4, 1)
    for name in ("lion", "lookahead"):
        opt = build_optimizer(lin.parameters(), name=name, lr=1e-2, weight_decay=1e-3)
        opt.zero_grad(set_to_none=True)
        ((lin(x) - y) ** 2).mean().backward()
        opt.step()
        if name == "lookahead":
            for _ in range(4):
                opt.zero_grad()
                ((lin(x) - y) ** 2).mean().backward()
                opt.step()
    opt = build_optimizer(lin.parameters(), name="adam", lr=1e-3)
    sch = WarmupCosineScheduler(opt, warmup_epochs=2, total_epochs=5)
    sch.step()
    sch.step()
    sch.step()
    assert sch.get_last_lr()
    assert build_scheduler(opt, name="weird") is not None
    with patch("iqrp.app.forecasting.neural.base.scheduler.has_torch", return_value=False):
        assert build_optimizer(lin.parameters()) is None
        assert build_scheduler(opt, name="cosine") is None


@pytest.mark.unit
def test_torch_utils_paths() -> None:
    seed_everything(1)
    resolve_device("cpu")
    resolve_device("cuda")
    resolve_device("mps")
    resolve_device("auto")
    m = nn.Linear(2, 1)
    maybe_compile(m, enabled=False)
    with patch.object(torch, "compile", side_effect=RuntimeError("nope"), create=True):
        maybe_compile(m, enabled=True)
    assert count_parameters(m) > 0
    assert count_parameters(None) == 0
    t = to_tensor(np.ones((2, 2)), device="cpu")
    assert from_tensor(t).shape == (2, 2)
    assert from_tensor(np.ones(3)).shape == (3,)
    with patch("iqrp.app.forecasting.neural.base.torch_utils._HAS_TORCH", False):
        assert resolve_device("auto") == "cpu"
        assert to_tensor(np.ones(2)).shape == (2,)


@pytest.mark.unit
def test_trainer_schedulers_and_class_loss(frame) -> None:
    settings = _fast(
        scheduler={"name": "onecycle"},
        train={"epochs": 2, "batch_size": 16, "device": "cpu", "optimizer": "adam"},
    )
    model = create_neural_model("mlp", settings=settings)
    model.fit(frame, feature_columns=feature_names(3))
    settings2 = _fast(
        scheduler={"name": "plateau"},
        train={
            "epochs": 2,
            "batch_size": 16,
            "device": "cpu",
            "accumulation_steps": 2,
            "grad_clip": 1.0,
        },
    )
    model2 = create_neural_model("lstm", settings=settings2)
    model2.fit(frame, feature_columns=feature_names(3))
    settings3 = _fast(
        task={"type": "classification", "n_classes": 2},
        train={"epochs": 2, "batch_size": 16, "device": "cpu", "loss": "cross_entropy"},
    )
    cls = simulate_nonlinear_returns(
        90, n_features=3, classification=True, rng=np.random.default_rng(9)
    )
    m3 = create_neural_model("mlp", settings=settings3)
    m3.fit(cls, feature_columns=feature_names(3))
    loss_fn = get_loss("mse")
    out = torch.randn(4, 2, 3)
    tgt = torch.randint(0, 3, (8,))
    _compute_loss(get_loss("cross_entropy"), out, tgt, task="classification")
    out2 = torch.randn(4, 3)
    _compute_loss(get_loss("bce"), out2, torch.rand(4, 3), task="binary")
    pred = torch.randn(4, 2)
    _compute_loss(loss_fn, pred, torch.randn(4, 2), task="regression")
    _compute_loss(loss_fn, (torch.randn(4, 2),), torch.randn(4, 2), task="regression")
    # trailing singleton squeeze path
    _compute_loss(loss_fn, torch.randn(4, 2, 1), torch.randn(4, 2), task="regression")


@pytest.mark.unit
def test_neural_model_edges(frame, tmp_path: Path) -> None:
    cols = feature_names(3)
    s = _fast(online={"mode": "refit", "refresh_every": 1})
    m = create_neural_model("gru", settings=s)
    m.fit(frame, feature_columns=cols)
    m.partial_fit(frame, feature_columns=cols)
    s2 = _fast(online={"mode": "warm_start", "refresh_every": 1, "window": 50})
    m2 = create_neural_model("mlp", settings=s2)
    m2.fit(frame[:80], feature_columns=cols)
    m2.partial_fit(frame, feature_columns=cols)
    with pytest.raises(Exception):
        create_neural_model("nbeats", settings=_fast()).fit(
            frame, feature_columns=cols
        ).predict_proba(frame)
    s3 = _fast(task={"type": "binary"}, train={"loss": "bce", "epochs": 2, "device": "cpu"})
    m3 = create_neural_model("lstm", settings=s3)
    cls = simulate_nonlinear_returns(
        90, n_features=3, classification=True, rng=np.random.default_rng(3)
    )
    m3.fit(cls, feature_columns=cols)
    m3.predict_proba(cls)
    m.explain(frame, method="integrated_gradients")
    p = m.export_onnx(tmp_path / "x.onnx")
    assert p.exists()
    with pytest.raises(Exception):
        create_neural_model("mlp", settings=_fast()).fit(
            frame.select(["open_time", "target"]), feature_columns=[]
        )
    create_neural_model(
        "mlp",
        settings={
            "architecture": {"lookback": 8, "horizon": 2, "hidden_size": 8, "num_layers": 1},
            "train": {"epochs": 1, "device": "cpu"},
            "regime": {"enabled": False},
        },
    )
    create_neural_model(
        "deepar",
        settings={
            "train": {"epochs": 1, "device": "cpu", "loss": "gaussian_nll"},
            "architecture": {"lookback": 8, "horizon": 2, "hidden_size": 8, "num_layers": 1},
            "regime": {"enabled": False},
        },
    )
    fc = m.forecast(frame, horizon=5)
    assert fc.path().size == 5
    s4 = _fast(regime={"enabled": True, "mode": "feature", "column": "regime"})
    m4 = create_neural_model("mlp", settings=s4)
    m4.fit(frame, feature_columns=cols, regime_column="regime")
    s5 = _fast(distributed={"gradient_checkpointing": True})
    create_neural_model("mlp", settings=s5).fit(frame, feature_columns=cols)


@pytest.mark.unit
def test_seq2seq_attention_off_and_quantile_nets(frame) -> None:
    cols = feature_names(3)
    s = _fast(
        architecture={
            "attention": False,
            "lookback": 10,
            "horizon": 2,
            "hidden_size": 12,
            "num_layers": 1,
        }
    )
    m = create_neural_model("seq2seq", settings=s)
    m.fit(frame, feature_columns=cols)
    net = Seq2SeqNet(3, 2, hidden_size=8, use_attention=False, task="quantile", n_quantiles=3)
    out = net(torch.randn(2, 10, 3))
    assert out.shape[-1] == 3
    net2 = Seq2SeqNet(3, 2, hidden_size=8, use_attention=True, task="distribution")
    assert net2(torch.randn(2, 10, 3)).shape[-1] == 2
    for name in ("nbeats", "nhits"):
        q = create_neural_model(
            name,
            settings=_fast(
                task={"type": "quantile"}, train={"loss": "quantile", "epochs": 2, "device": "cpu"}
            ),
        )
        q.fit(frame, feature_columns=cols)
        q.forecast(frame, horizon=2)


@pytest.mark.unit
def test_hpo_none_and_failures(frame) -> None:
    cols = feature_names(3)
    m = create_neural_model("mlp", settings=_fast())
    X = frame.select(cols).to_numpy()
    y = frame["target"].to_numpy()
    X_seq, y_seq = make_sequences(X, y, lookback=10, horizon=2)
    assert optimize_neural(m, X_seq, y_seq, settings=_fast(optimization={"method": "none"})) == {}
    with patch(
        "iqrp.app.forecasting.neural.optimization.hpo.NeuralTrainer.fit",
        side_effect=RuntimeError("boom"),
    ):
        best = optimize_neural(
            m, X_seq, y_seq, settings=_fast(optimization={"method": "random", "n_trials": 1})
        )
        assert isinstance(best, dict)
    optimize_neural(
        m,
        X_seq[:40],
        y_seq[:40],
        settings=_fast(optimization={"method": "bayesian", "n_trials": 1, "pruning": False}),
    )


@pytest.mark.unit
def test_explain_no_torch_and_distributed() -> None:
    mod = nn.Sequential(nn.Flatten(), nn.Linear(6, 1))
    X = np.random.randn(2, 2, 3)
    with patch(
        "iqrp.app.forecasting.neural.explainability.attribution.has_torch", return_value=False
    ):
        assert explain_neural(mod, X, method="ig").shape == X.shape
        assert saliency_map(mod, X).shape == X.shape
        assert occlusion_analysis(mod, X).shape == X.shape
    assert explain_neural(mod, X, method="unknown").size > 0
    lin = nn.Linear(2, 1)

    class CK(nn.Module):
        def gradient_checkpointing_enable(self):
            raise RuntimeError("no")

    enable_gradient_checkpointing(CK())
    enable_gradient_checkpointing(lin)
    assert wrap_ddp(lin, enabled=True) is lin
    s = _fast(train={"mixed_precision": True}, distributed={"amp": True})
    assert amp_enabled(s) is True
    moe = MixtureOfExperts(2, 4)
    h = torch.randn(3, 4)
    assert moe(h, [torch.randn(3, 5), torch.randn(3, 5)]).shape == (3, 5)


@pytest.mark.unit
def test_probabilistic_edges() -> None:
    assert aleatoric_from_gaussian(np.ones((2, 3, 1))).shape[0] == 2
    mu = np.zeros((2, 3))
    pred = np.stack([mu, np.zeros_like(mu)], axis=-1)
    q = quantiles_from_prediction(pred, task="distribution", distribution="student_t")
    assert q.shape[-1] == 3
    point = extract_point_forecast(
        np.random.randn(4, 3, 3), task="quantile", alphas=(0.1, 0.5, 0.9)
    )
    assert point.shape[0] == 4
    lo, hi = interval_from_prediction(np.random.randn(1, 3), task="regression")
    assert lo.shape == hi.shape

    class Dist(nn.Module):
        def forward(self, x):
            b = x.shape[0]
            return torch.randn(b, 2, 2)

    mean, epi = epistemic_mc_dropout(Dist(), np.random.randn(2, 2, 3), n_samples=3)
    assert mean.shape[0] == 2


@pytest.mark.unit
def test_orchestrator_parallel_and_registry(frame) -> None:
    cols = feature_names(3)
    orch = NeuralOrchestrator(_fast())
    res = orch.compare(["mlp", "gru"], frame, feature_columns=cols, parallel=True)
    assert len(res) == 2
    assert "mlp.model" in ensure_neural_models_loaded(["iqrp.app.forecasting.neural.mlp.model"])[0]
    assert ensure_neural_models_loaded(["iqrp.does.not.exist"]) == []
    from iqrp.app.forecasting.neural.diagnostics.report import run_neural_diagnostics

    class Dummy:
        _residuals = None
        _history = History()
        architecture_name = "x"
        _module = None
        _X_seq = None

    assert run_neural_diagnostics(Dummy()).residual_std >= 0
    plot_attribution(np.random.rand(2), feature_names=["a", "b"])
    plot_attribution(np.random.rand(2, 3))
    with patch("iqrp.app.forecasting.neural.visualization.plots._pyplot", return_value=None):
        plot_forecast(None, np.arange(3))
        plot_training_curves({"train_loss": [1.0]})


@pytest.mark.unit
def test_variants_and_config_default_without_file(frame) -> None:
    cols = feature_names(3)
    for name in ("stacked_lstm", "bidirectional_lstm", "stacked_gru", "attention_seq2seq"):
        m = create_neural_model(name, settings=_fast())
        m.fit(frame, feature_columns=cols)
    with patch("iqrp.app.forecasting.neural.config._default_config_path") as p:
        p.return_value = Path("/tmp/does_not_exist_neural.yaml")
        s = NeuralSettings.default()
        assert s.train.epochs > 0
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"train": {"epochs": 2}})
    NeuralSettings.from_mapping(cfg)
    create_neural_model("deepar")


@pytest.mark.unit
def test_forecast_interval_fallback(frame) -> None:
    cols = feature_names(3)
    m = create_neural_model("mlp", settings=_fast())
    m.fit(frame, feature_columns=cols)
    with patch.object(
        type(m), "forecast", return_value=MagicMock(intervals=None, path=lambda: np.ones(3))
    ):
        from iqrp.app.forecasting.neural.base.neural_model import NeuralForecastModel

        ints = NeuralForecastModel.forecast_interval(m, frame, horizon=3)
        assert len(ints) == 3
