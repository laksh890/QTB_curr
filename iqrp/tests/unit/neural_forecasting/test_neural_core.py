"""Core unit tests for Institutional Neural Forecasting Platform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.forecasting.neural import (
    NeuralOrchestrator,
    NeuralSettings,
    create_neural_model,
    list_neural_models,
)
from iqrp.app.forecasting.neural.base.losses import get_loss, register_custom_loss
from iqrp.app.forecasting.neural.base.metrics import evaluate_predictions
from iqrp.app.forecasting.neural.base.processes import feature_names, multi_horizon_frame, simulate_nonlinear_returns
from iqrp.app.forecasting.neural.base.scheduler import build_optimizer, build_scheduler
from iqrp.app.forecasting.neural.base.torch_utils import has_torch, seed_everything
from iqrp.app.forecasting.neural.embeddings import (
    CategoricalEmbedding,
    MixtureOfExperts,
    PositionalEncoding,
    RegimeEmbedding,
    RegimeGate,
    TemporalEmbedding,
)
from iqrp.app.forecasting.neural.explainability.attribution import explain_neural
from iqrp.app.forecasting.neural.optimization.distributed import amp_enabled, enable_gradient_checkpointing, wrap_ddp
from iqrp.app.forecasting.neural.optimization.hpo import optimize_neural
from iqrp.app.forecasting.neural.probabilistic import (
    gaussian_quantiles,
    sample_gaussian,
    student_t_quantiles,
    total_uncertainty,
)
from iqrp.app.forecasting.neural.visualization.plots import (
    plot_attention,
    plot_attribution,
    plot_forecast,
    plot_loss_curve,
    plot_residual_distribution,
    plot_training_curves,
)


def _fast_settings(**extra: object) -> NeuralSettings:
    base = {
        "architecture": {"lookback": 12, "horizon": 3, "hidden_size": 16, "num_layers": 1, "n_blocks": 1, "dropout": 0.0},
        "train": {"epochs": 2, "batch_size": 32, "device": "cpu", "early_stopping_patience": 20, "seed": 0},
        "scheduler": {"name": "none"},
        "regime": {"enabled": False},
        "optimization": {"method": "none"},
        "visualization": {"enabled": False},
        "online": {"finetune_epochs": 1, "window": 80, "refresh_every": 1000},
    }
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}  # type: ignore[dict-item]
        else:
            base[k] = v  # type: ignore[assignment]
    return NeuralSettings.from_mapping(base)


@pytest.fixture
def reg_frame() -> pl.DataFrame:
    return simulate_nonlinear_returns(160, n_features=4, rng=np.random.default_rng(1))


@pytest.fixture
def cls_frame() -> pl.DataFrame:
    return simulate_nonlinear_returns(160, n_features=4, classification=True, rng=np.random.default_rng(2))


@pytest.mark.unit
def test_torch_available() -> None:
    assert has_torch()


@pytest.mark.unit
def test_registry_lists_all_models() -> None:
    names = set(list_neural_models())
    assert names >= {
        "mlp",
        "lstm",
        "stacked_lstm",
        "bidirectional_lstm",
        "gru",
        "stacked_gru",
        "tcn",
        "nbeats",
        "nhits",
        "deepar",
        "seq2seq",
        "attention_seq2seq",
    }


@pytest.mark.unit
def test_settings_hydra() -> None:
    s = NeuralSettings.default()
    assert s.architecture.hidden_size > 0
    s2 = NeuralSettings.from_mapping({"task": {"type": "quantile"}, "train": {"loss": "quantile"}})
    assert s2.task.type == "quantile"
    s3 = NeuralSettings.from_hydra(overrides=["forecast.default_horizon=7"])
    assert s3.forecast.default_horizon == 7
    with pytest.raises(Exception):
        NeuralSettings.from_mapping({"task": {"type": "not_a_task"}})


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        "mlp",
        "lstm",
        "stacked_lstm",
        "bidirectional_lstm",
        "gru",
        "stacked_gru",
        "tcn",
        "nbeats",
        "nhits",
        "deepar",
        "seq2seq",
        "attention_seq2seq",
    ],
)
def test_all_models_api(name: str, reg_frame: pl.DataFrame) -> None:
    settings = _fast_settings()
    if name == "deepar":
        settings = _fast_settings(
            task={"type": "distribution"},
            train={"loss": "gaussian_nll", "epochs": 2, "batch_size": 32, "device": "cpu"},
            probabilistic={"enabled": True, "distribution": "gaussian"},
        )
    model = create_neural_model(name, settings=settings)
    cols = feature_names(4)
    model.fit(reg_frame, feature_columns=cols)
    pred = model.predict(reg_frame)
    assert pred.shape[0] == reg_frame.height
    fc = model.forecast(reg_frame, horizon=3)
    assert fc.path().shape == (3,)
    assert len(model.forecast_interval(reg_frame, horizon=3)) == 3
    report = model.evaluate(reg_frame)
    assert "rmse" in report.metrics or "mae" in report.metrics
    expl = model.explain(reg_frame, method="gradient")
    assert expl.importances
    diag = model.diagnostics()
    assert "residual_std" in diag


@pytest.mark.unit
def test_classification_proba(cls_frame: pl.DataFrame) -> None:
    settings = _fast_settings(task={"type": "binary"}, train={"loss": "bce", "epochs": 2, "device": "cpu"})
    model = create_neural_model("mlp", settings=settings)
    cols = feature_names(4)
    model.fit(cls_frame, feature_columns=cols)
    proba = model.predict_proba(cls_frame)
    assert proba.shape[0] == cls_frame.height
    assert proba.shape[1] == 2


@pytest.mark.unit
def test_quantile_and_partial_fit(reg_frame: pl.DataFrame) -> None:
    cols = feature_names(4)
    settings = _fast_settings(
        task={"type": "quantile", "quantile_alphas": [0.1, 0.5, 0.9]},
        train={"loss": "quantile", "epochs": 2, "device": "cpu"},
        online={"mode": "finetune", "finetune_epochs": 1, "window": 60, "refresh_every": 1000},
    )
    model = create_neural_model("lstm", settings=settings)
    model.fit(reg_frame[:120], feature_columns=cols)
    model.partial_fit(reg_frame, feature_columns=cols)
    fc = model.forecast(reg_frame, horizon=3)
    assert fc.path().size == 3


@pytest.mark.unit
def test_save_load_export(tmp_path: Path, reg_frame: pl.DataFrame) -> None:
    cols = feature_names(4)
    model = create_neural_model("gru", settings=_fast_settings())
    model.fit(reg_frame, feature_columns=cols)
    path = tmp_path / "gru.json"
    model.save(path)
    loaded = type(model).load(path)
    assert loaded.evaluate(reg_frame).metrics["n"] > 0
    onnx_path = model.export_onnx(tmp_path / "gru.onnx")
    assert onnx_path.exists()


@pytest.mark.unit
def test_losses_schedulers_metrics() -> None:
    import torch

    seed_everything(0)
    y = torch.randn(8, 3)
    for name in ("mse", "mae", "huber", "logcosh", "quantile", "gaussian_nll", "student_t_nll", "focal", "bce"):
        fn = get_loss(name, alphas=(0.1, 0.5, 0.9))
        if name == "quantile":
            pred = torch.randn(8, 3, 3)
            assert float(fn(pred, y).item()) >= 0
        elif name in {"gaussian_nll", "student_t_nll"}:
            pred = torch.randn(8, 3, 2)
            assert float(fn(pred, y).item()) == float(fn(pred, y).item())
        elif name in {"focal", "bce"}:
            pred = torch.randn(8)
            tgt = (torch.rand(8) > 0.5).float()
            assert float(fn(pred, tgt).item()) >= 0
        else:
            pred = torch.randn(8, 3)
            assert float(fn(pred, y).item()) >= 0
    register_custom_loss("my_mse", lambda p, t: ((p - t) ** 2).mean())
    assert float(get_loss("my_mse")(y, y).item()) == 0.0
    lin = torch.nn.Linear(4, 2)
    for opt_name in ("adam", "adamw", "rmsprop", "sgd", "lion", "lookahead"):
        opt = build_optimizer(lin.parameters(), name=opt_name, lr=1e-3)
        assert opt is not None
    opt = build_optimizer(lin.parameters(), name="adam", lr=1e-3)
    for sched_name in ("cosine", "onecycle", "plateau", "warmup_cosine", "exponential", "none"):
        sch = build_scheduler(opt, name=sched_name, epochs=5, steps_per_epoch=2)
        if sched_name != "none":
            assert sch is not None
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.1, 1.9, 3.2])
    m = evaluate_predictions(yt, yp, task="regression")
    assert "rmse" in m


@pytest.mark.unit
def test_embeddings_and_moe() -> None:
    import torch

    cat = CategoricalEmbedding(5, 8)
    assert cat(torch.tensor([0, 1, 2])).shape == (3, 8)
    reg = RegimeEmbedding(4, 8)
    assert reg(torch.tensor([0, 1])).shape == (2, 8)
    gate = RegimeGate(8, 4)
    h = torch.randn(2, 8)
    onehot = torch.eye(4)[:2]
    assert gate(h, onehot).shape == (2, 8)
    moe = MixtureOfExperts(3, 8)
    experts = torch.randn(2, 3, 5)
    assert moe(h, experts).shape == (2, 5)
    pos = PositionalEncoding(8)
    assert pos(torch.randn(2, 10, 8)).shape == (2, 10, 8)
    temp = TemporalEmbedding(8)
    assert temp(torch.randn(2, 10, 3)).shape[0] == 2


@pytest.mark.unit
def test_probabilistic_and_viz() -> None:
    mu = np.zeros((4, 3))
    sig = np.ones((4, 3)) * 0.5
    q = gaussian_quantiles(mu, sig)
    assert q.shape[-1] == 3
    qt = student_t_quantiles(mu, sig)
    assert qt.shape == q.shape
    samples = sample_gaussian(mu, sig, n=10)
    assert samples.shape[0] == 10
    plot_forecast(np.arange(5), np.arange(5) + 0.1, intervals=(np.arange(5) - 1, np.arange(5) + 1))
    plot_training_curves({"train_loss": [1.0, 0.5], "val_loss": [1.1, 0.6]})
    plot_attention(np.random.rand(3, 5))
    plot_attribution(np.random.rand(2, 4, 3), feature_names=["a", "b", "c"])
    plot_residual_distribution(np.random.randn(50))
    plot_loss_curve([1.0, 0.8, 0.5])


@pytest.mark.unit
def test_hpo_and_orchestrator(reg_frame: pl.DataFrame) -> None:
    cols = feature_names(4)
    settings = _fast_settings(optimization={"method": "random", "n_trials": 2})
    model = create_neural_model("mlp", settings=settings)
    model.fit(reg_frame, feature_columns=cols)
    assert model.is_fitted
    orch = NeuralOrchestrator(_fast_settings())
    m2, result = orch.fit("lstm", reg_frame, feature_columns=cols)
    assert result.metrics
    assert m2.is_fitted
    compared = orch.compare(["mlp", "gru"], reg_frame, feature_columns=cols)
    assert set(compared) == {"mlp", "gru"}


@pytest.mark.unit
def test_regime_separate_and_distributed(reg_frame: pl.DataFrame) -> None:
    cols = feature_names(4)
    settings = _fast_settings(regime={"enabled": True, "column": "regime", "mode": "separate"})
    model = create_neural_model("mlp", settings=settings)
    model.fit(reg_frame, feature_columns=cols, regime_column="regime")
    pred = model.predict(reg_frame)
    assert pred.shape[0] == reg_frame.height
    import torch

    lin = torch.nn.Linear(4, 2)
    assert wrap_ddp(lin, enabled=False) is lin
    assert wrap_ddp(lin, enabled=True) is lin
    enable_gradient_checkpointing(lin)
    assert amp_enabled(settings) is False


@pytest.mark.unit
def test_explain_methods(reg_frame: pl.DataFrame) -> None:
    cols = feature_names(4)
    model = create_neural_model("tcn", settings=_fast_settings())
    model.fit(reg_frame, feature_columns=cols)
    X = model._last_window(reg_frame, cols)
    for method in ("integrated_gradients", "saliency", "occlusion", "shap"):
        attr = explain_neural(model._module, X, method=method, device=model._device, steps=4)
        assert np.asarray(attr).size > 0


@pytest.mark.unit
def test_multi_horizon_frame() -> None:
    frame = multi_horizon_frame(80, horizon=3, rng=np.random.default_rng(3))
    assert "target_h1" in frame.columns


@pytest.mark.unit
def test_optimize_neural_direct(reg_frame: pl.DataFrame) -> None:
    cols = feature_names(4)
    settings = _fast_settings()
    model = create_neural_model("mlp", settings=settings)
    # build sequences like fit
    from iqrp.app.forecasting.neural.base.data import make_sequences, standardize_apply, standardize_fit

    X = reg_frame.select(cols).to_numpy()
    y = reg_frame["target"].to_numpy()
    mu, sd = standardize_fit(X)
    Xs = standardize_apply(X, mu, sd)
    X_seq, y_seq = make_sequences(Xs, y, lookback=12, horizon=3)
    best = optimize_neural(model, X_seq, y_seq, settings=_fast_settings(optimization={"method": "grid", "n_trials": 2}))
    assert isinstance(best, dict)
    best2 = optimize_neural(
        model, X_seq, y_seq, settings=_fast_settings(optimization={"method": "optuna", "n_trials": 2})
    )
    assert isinstance(best2, dict)
