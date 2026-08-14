"""Core unit tests for the Institutional Forecasting Framework."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.core.exceptions import ConfigurationError, ValidationError
from iqrp.app.forecasting import (
    Forecast,
    ForecastEvaluator,
    ForecastingPipeline,
    ForecastingSettings,
    ForecastModelMeta,
    ForecastScheduler,
    ForecastTrainer,
    Prediction,
    PredictionInterval,
    get_registry,
    register_forecast_model,
)
from iqrp.app.forecasting.base.forecast_model import ForecastModel
from iqrp.app.forecasting.base.metadata import TrainingMetadata
from iqrp.app.forecasting.base.prediction import DistributionForecast, QuantileForecast
from iqrp.app.forecasting.base.registry import ensure_forecast_models_loaded, forecast_model_factory
from iqrp.app.forecasting.diagnostics import (
    calibration_report,
    detect_bias,
    detect_feature_drift,
    detect_prediction_drift,
    forecast_error_by_horizon,
    residual_analysis,
)
from iqrp.app.forecasting.explainability import (
    attribute,
    attribution_matrix,
    compare_attributions,
    explain_model,
    top_k_features,
)
from iqrp.app.forecasting.models.mock import MockForecastModel
from iqrp.app.forecasting.orchestration.inference import (
    StreamingInference,
    batch_forecast,
    batch_predict,
)
from iqrp.app.forecasting.postprocessing import (
    ProbabilityCalibrator,
    confidence_intervals_from_samples,
    distribution_from_samples,
    forecast_uncertainty_report,
    gaussian_intervals,
    quantile_from_samples,
    quantile_intervals,
    residual_intervals,
)
from iqrp.app.forecasting.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    Scaler,
    encode_frame_categoricals,
    make_windows,
    recursive_path,
    select_features,
)
from iqrp.app.forecasting.visualization import (
    plot_feature_importance,
    plot_forecast,
    plot_horizon_comparison,
    plot_residuals,
    plot_rolling_accuracy,
)


def _frame(n: int = 60, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    f0 = np.linspace(0, 1, n) + rng.normal(0, 0.05, n)
    f1 = np.sin(t / 5) + rng.normal(0, 0.05, n)
    target = 0.5 * f0 + 0.3 * f1 + rng.normal(0, 0.02, n)
    regime = (t > n // 2).astype(np.int64)
    return pl.DataFrame(
        {
            "open_time": t.tolist(),
            "f0": f0,
            "f1": f1,
            "cat": ["a" if i % 2 == 0 else "b" for i in range(n)],
            "target": target,
            "regime": regime,
        }
    )


@pytest.mark.unit
def test_registry_and_meta() -> None:
    ensure_forecast_models_loaded()
    assert "mock" in get_registry().list_names()
    meta = get_registry().describe("mock")
    assert meta.name == "mock"
    assert get_registry().all_meta()
    factory = forecast_model_factory("mock")
    assert factory is MockForecastModel
    m = get_registry().create("mock")
    assert isinstance(m, MockForecastModel)
    d = meta.to_dict()
    assert ForecastModelMeta.from_dict(d).name == "mock"
    tm = TrainingMetadata(1, 2, ("a",), "t", None, 1)
    assert TrainingMetadata.from_dict(tm.to_dict()).n_samples == 1


@pytest.mark.unit
def test_registry_errors() -> None:
    with pytest.raises(ConfigurationError):
        get_registry().get_class("does_not_exist")

    class Bad:
        pass

    with pytest.raises(ConfigurationError):
        get_registry().register(Bad)  # type: ignore[arg-type]


@pytest.mark.unit
def test_mock_fit_predict_forecast_roundtrip(tmp_path: Path) -> None:
    frame = _frame()
    model = MockForecastModel(settings=ForecastingSettings.default())
    model.fit(frame, ["f0", "f1"], target_column="target", regime_column="regime")
    assert model.is_fitted
    pred = model.predict(frame)
    assert pred.shape[0] == frame.height
    proba = model.predict_proba(frame)
    assert proba.shape == (frame.height, 3)
    fc = model.forecast(frame, horizon=5)
    assert fc.values.shape[0] == 5
    assert fc.intervals is not None
    assert fc.one_step().horizon == 1
    assert fc.n_step(3).horizon == 3
    intervals = model.forecast_interval(frame, horizon=3, level=0.9)
    assert len(intervals) == 3
    report = model.evaluate(frame, feature_columns=["f0", "f1"], target_column="target")
    assert "rmse" in report.metrics
    expl = model.explain(frame, ["f0", "f1"], method="permutation")
    assert "f0" in expl.importances
    path = tmp_path / "mock.json"
    model.save(path)
    loaded = MockForecastModel.load(path)
    assert loaded.is_fitted
    assert np.allclose(loaded.predict(frame), pred)
    ck = model.checkpoint()
    model2 = MockForecastModel()
    model2.restore_checkpoint(ck)
    assert model2.is_fitted


@pytest.mark.unit
def test_partial_fit_and_online() -> None:
    frame = _frame(40)
    model = MockForecastModel()
    model.partial_fit(frame.slice(0, 20), ["f0", "f1"], target_column="target")
    model.partial_fit(frame.slice(20, 20), ["f0", "f1"], target_column="target")
    assert model.is_fitted


@pytest.mark.unit
def test_forecast_objects() -> None:
    fc = Forecast.from_values(
        [1.0, 2.0, 3.0], horizon=3, timestamps=(1, 2, 3), probabilities=np.eye(3)
    )
    assert fc.point(2).value == 2.0
    d = fc.to_dict()
    assert d["horizon"] == 3
    with pytest.raises(IndexError):
        fc.point(10)
    p = Prediction(value=1.0, horizon=1, probability=np.array([0.1, 0.9]))
    assert "value" in p.to_dict()
    pi = PredictionInterval(0.0, 1.0, level=0.95)
    assert float(pi.width) == 1.0
    assert "lower" in pi.to_dict()
    q = QuantileForecast({0.1: 0.0, 0.9: 1.0}, horizon=1)
    assert "quantiles" in q.to_dict()
    dist = DistributionForecast(mean=0.0, variance=1.0, samples=np.ones(5))
    assert dist.to_dict()["mean"] == 0.0
    fc2 = Forecast.from_values(1.0, regime_used=np.array([1, 2]))
    assert fc2.point(1).regime in {1, 2}


@pytest.mark.unit
def test_evaluator_metrics_and_cv() -> None:
    y = np.linspace(0, 1, 40)
    pred = y + 0.01
    ev = ForecastEvaluator()
    r = ev.evaluate(y, pred, task="regression")
    assert r.metrics["mae"] < 0.1
    labels = (y > 0.5).astype(int)
    scores = pred
    c = ev.evaluate_classification(labels, labels, scores=scores)
    assert c["accuracy"] == 1.0
    proba = np.column_stack([1 - scores, scores])
    p = ev.evaluate_probability(proba, labels)
    assert "brier" in p
    wf = ev.cross_validate(y, pred, method="walk_forward", train_size=10, test_size=2, step=5)
    assert wf.folds
    roll = ev.cross_validate(y, pred, method="rolling", window=10, test_size=1, step=5)
    assert roll.method == "rolling"
    ts = ev.cross_validate(y, pred, method="time_series_split", n_splits=3)
    assert ts.folds
    board = ev.benchmark({"a": {"rmse": 0.2}, "b": {"rmse": 0.1}}, primary="rmse")
    assert board[0]["model"] == "b"


@pytest.mark.unit
def test_preprocessing_postprocessing() -> None:
    x = np.random.default_rng(0).normal(size=(50, 3))
    for kind in ("none", "standard", "minmax", "robust"):
        sc = Scaler(kind=kind).fit(x)  # type: ignore[arg-type]
        xt = sc.transform(x)
        assert Scaler.from_dict(sc.to_dict()).fitted
        _ = sc.inverse_transform(xt)
    le = LabelEncoder().fit(["a", "b", "a"])
    assert le.transform(["b", "a"]).tolist() == [1, 0]
    assert le.inverse_transform(np.array([0, 1])) == ["a", "b"]
    oh = OneHotEncoder().fit(["a", "b"])
    assert oh.fit_transform(["a", "b"]).shape[1] == 2
    frame = _frame(20)
    enc_frame, encs = encode_frame_categoricals(frame, ["cat"])
    assert "cat" in encs
    wb = make_windows(x, x[:, 0], window_size=5, horizon=2, flatten=True)
    assert wb.X.shape[0] > 0
    path = recursive_path(x[:5], lambda w: float(w[-1, 0]), horizon=3)
    assert path.shape[0] == 3
    idx = select_features(x, x[:, 0], method="variance", max_features=2)
    assert idx.size <= 2
    idx2 = select_features(x, x[:, 0], method="correlation", max_features=2)
    assert idx2.size <= 2
    idx3 = select_features(x, x[:, 0], method="mutual_info", max_features=2)
    assert idx3.size <= 2
    cal = ProbabilityCalibrator(method="temperature").fit(np.eye(10, 3) + 0.1, np.arange(10) % 3)
    _ = cal.transform(np.eye(5, 3) + 0.05)
    assert ProbabilityCalibrator.from_dict(cal.to_dict()).fitted
    for method in ("platt", "isotonic", "none"):
        ProbabilityCalibrator(method=method).fit(  # type: ignore[arg-type]
            np.random.default_rng(0).random((20, 3)), np.arange(20) % 3
        ).transform(np.random.default_rng(1).random((5, 3)))
    ints = residual_intervals(np.array([1.0, 2.0]), level=0.95)
    assert len(ints) == 2
    g = gaussian_intervals(np.array([0.0, 1.0]), np.array([1.0, 1.0]))
    assert g[0].kind == "prediction"
    qi = quantile_intervals({0.05: np.array([0.0]), 0.5: np.array([1.0]), 0.95: np.array([2.0])})
    assert qi[0].upper == 2.0
    ci = confidence_intervals_from_samples(np.random.default_rng(0).normal(size=(100, 3)))
    assert len(ci) == 3
    qf = quantile_from_samples(np.random.default_rng(0).normal(size=(50, 2)))
    assert len(qf) == 2
    dist = distribution_from_samples(np.ones(10))
    assert dist.mean == 1.0
    ur = forecast_uncertainty_report(
        np.array([1.0, 2.0]), intervals_width=np.array([0.1, 0.2]), probabilities=np.eye(2)
    )
    assert "mean_entropy" in ur


@pytest.mark.unit
def test_diagnostics_explain_viz(tmp_path: Path) -> None:
    y = np.linspace(0, 1, 30)
    pred = y + 0.05
    rr = residual_analysis(y, pred)
    assert "mae" in rr.to_dict()
    assert forecast_error_by_horizon(y, np.column_stack([pred, pred]), horizons=2)
    dr = detect_prediction_drift(pred[:15], pred[15:], method="psi")
    assert "score" in dr.to_dict()
    detect_prediction_drift(pred[:15], pred[15:], method="ks")
    detect_prediction_drift(pred[:15], pred[15:] + 2, method="mean_shift")
    fd = detect_feature_drift(
        np.random.default_rng(0).normal(size=(40, 2)), np.random.default_rng(1).normal(size=(40, 2))
    )
    assert fd.method == "psi_features"
    proba = np.column_stack([1 - pred, pred])
    labels = (y > 0.5).astype(int)
    cr = calibration_report(proba, labels)
    assert cr.ece >= 0
    assert "bias" in detect_bias(y, pred)
    frame = _frame(40)
    model = MockForecastModel().fit(frame, ["f0", "f1"], target_column="target")
    for method in ("permutation", "builtin", "shap", "integrated_gradients", "attention"):
        expl = explain_model(model, frame, ["f0", "f1"], method=method)
        assert expl.method
    a = attribute(model, frame, ["f0", "f1"], method="shap")
    assert top_k_features(a, 1)
    assert attribution_matrix(a).size
    b = explain_model(model, frame, ["f0", "f1"], method="builtin")
    assert compare_attributions(a, b)
    plot_forecast(y, pred, tmp_path / "f.svg", lower=pred - 0.1, upper=pred + 0.1)
    plot_residuals(rr.residuals, tmp_path / "r.svg")
    plot_rolling_accuracy(np.linspace(0.5, 0.9, 20), tmp_path / "a.svg")
    plot_feature_importance({"f0": 0.7, "f1": 0.3}, tmp_path / "i.svg")
    plot_horizon_comparison({1: 0.1, 2: 0.2}, tmp_path / "h.svg")


@pytest.mark.unit
def test_trainer_pipeline_scheduler_inference(tmp_path: Path) -> None:
    frame = _frame(50)
    settings = ForecastingSettings.from_mapping(
        {
            **ForecastingSettings.default().model_dump(),
            "columns": {
                "timestamp": "open_time",
                "target": "target",
                "feature_columns": ("f0", "f1"),
                "regime_column": "regime",
            },
            "preprocessing": {
                **ForecastingSettings.default().preprocessing.model_dump(),
                "feature_selection": "variance",
                "max_features": 2,
            },
            "online": {
                "warm_start": True,
                "rolling_retrain_every": 2,
                "checkpoint_every": 1,
                "stream_buffer": 10,
            },
        }
    )
    model = MockForecastModel(settings=settings)
    tr = ForecastTrainer(settings).fit(
        model, frame, feature_columns=["f0", "f1"], target_column="target"
    )
    assert tr.training.n_samples == 50
    ForecastTrainer(settings).partial_fit(
        model, frame.slice(40, 10), feature_columns=["f0", "f1"], target_column="target"
    )
    pipe = ForecastingPipeline(settings=settings, model_name="mock")
    result = pipe.run(frame, horizon=4)
    assert result.forecast.horizon == 4
    sched = ForecastScheduler(settings)
    actions = sched.on_update(
        MockForecastModel(settings=settings),
        frame.slice(0, 15),
        feature_columns=["f0", "f1"],
        target_column="target",
    )
    assert actions["retrained"]
    sched.on_update(
        model, frame.slice(15, 10), feature_columns=["f0", "f1"], target_column="target"
    )
    sched.reset()
    stream = StreamingInference(model=model, settings=settings)
    for i in range(5):
        stream.push({"f0": float(i), "f1": float(i) * 0.1, "target": float(i)})
    assert stream.forecast(horizon=2).values.size == 2
    assert (
        batch_predict(model, frame, feature_columns=["f0", "f1"], batch_size=7).shape[0]
        == frame.height
    )
    assert batch_forecast(model, frame, horizon=2, feature_columns=["f0", "f1"]).horizon == 2
    settings2 = ForecastingSettings.from_hydra(overrides=["inference.default_horizon=7"])
    assert settings2.inference.default_horizon == 7


@pytest.mark.unit
def test_not_fitted_and_no_proba_model() -> None:
    m = MockForecastModel()
    with pytest.raises(ValidationError):
        m.predict(_frame(5))
    with pytest.raises(ValidationError):
        m.checkpoint()
    with pytest.raises(ValidationError):
        m.restore_checkpoint()

    @register_forecast_model
    class NoProba(ForecastModel):
        meta = ForecastModelMeta(
            name="no_proba_test",
            version="0",
            description="x",
            algorithm_family="x",
            supports_proba=False,
        )

        def fit(self, frame, feature_columns=None, *, target_column=None, regime_column=None):
            self._fitted = True
            self._feature_columns = feature_columns or []
            return self

        def predict(self, frame, feature_columns=None):
            return np.zeros(frame.height)

        def forecast(self, frame, *, horizon=None, feature_columns=None):
            return Forecast.from_values(np.zeros(horizon or 1))

        def _algorithm_state(self):
            return {}

        def _load_algorithm_state(self, state):
            return None

    np_model = NoProba()
    np_model.fit(_frame(5), ["f0"])
    with pytest.raises(ValidationError):
        np_model.predict_proba(_frame(5))
