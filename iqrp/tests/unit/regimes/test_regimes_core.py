"""Unit tests for regime framework primitives and mock model."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.core.exceptions import ConfigurationError, ValidationError
from iqrp.app.regimes import (
    MockRegimeModel,
    PersistenceEngine,
    ProbabilityEngine,
    RegimeDetector,
    RegimeEvaluator,
    RegimeForecast,
    RegimeModelMeta,
    RegimeModelRegistry,
    RegimeSerializer,
    RegimeSettings,
    RegimeState,
    RegimeStore,
    RegimeTrainer,
    RegimeTransition,
    ensure_regime_models_loaded,
    get_registry,
    plot_persistence,
    plot_probabilities,
    plot_timeline,
    plot_transitions,
    regime_model_factory,
    register_regime_model,
)
from iqrp.app.regimes.base.regime_model import RegimeModel


def _ohlcv(n: int = 160, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rets = rng.normal(0.0005, 0.012, size=n)
    close = 100 * np.cumprod(1 + rets)
    rows = []
    for i in range(n):
        c = float(close[i])
        rows.append(
            {
                "open_time": start + timedelta(hours=i),
                "open": c * (1 - 0.001),
                "high": c * (1 + 0.01),
                "low": c * (1 - 0.01),
                "close": c,
                "volume": float(10 + (i % 7)),
                "feat_a": float(rets[i]),
            }
        )
    return pl.DataFrame(rows)


@pytest.mark.unit
def test_settings_hydra_and_defaults() -> None:
    settings = RegimeSettings.from_hydra(overrides=["detection.confidence_threshold=0.7"])
    assert settings.detection.confidence_threshold == 0.7
    assert settings.default_model == "mock_regime"
    assert RegimeSettings.default().enabled is True
    mapped = RegimeSettings.from_mapping({"enabled": True, "default_model": "mock_regime"})
    assert mapped.default_model == "mock_regime"
    with pytest.raises(ConfigurationError):
        RegimeSettings.from_mapping("not-a-mapping")  # type: ignore[arg-type]


@pytest.mark.unit
def test_registry_discovery() -> None:
    ensure_regime_models_loaded()
    reg = get_registry()
    assert "mock_regime" in reg.list_names()
    meta = reg.describe("mock_regime")
    assert meta.algorithm_family == "mock"
    assert meta.to_dict()["n_states"] == 3
    assert reg.all_meta()
    factory = regime_model_factory("mock_regime")
    assert issubclass(factory, RegimeModel)
    with pytest.raises(ConfigurationError):
        reg.get_class("does_not_exist")


@pytest.mark.unit
def test_registry_rejects_missing_meta() -> None:
    local = RegimeModelRegistry()

    class Bad:
        pass

    with pytest.raises(ConfigurationError):
        local.register(Bad)  # type: ignore[arg-type]


@pytest.mark.unit
def test_state_and_transition_roundtrip() -> None:
    ts = RegimeState.now_utc()
    state = RegimeState(
        state_id=1,
        state_name="sideways",
        probability=0.7,
        confidence=0.7,
        persistence=0.8,
        start_time=ts,
        end_time=ts,
        duration=3.0,
        features_used=("close",),
        model_version="1.0.0",
        timestamp=ts,
        metadata={"k": 1},
    )
    restored = RegimeState.from_dict(state.to_dict())
    assert restored.state_id == 1
    assert restored.features_used == ("close",)

    tr = RegimeTransition(
        previous_state=0,
        current_state=2,
        probability=0.2,
        confidence=0.2,
        timestamp=ts,
        previous_name="bear",
        current_name="bull",
    )
    assert RegimeTransition.from_dict(tr.to_dict()).current_state == 2


@pytest.mark.unit
def test_probability_engine() -> None:
    tm = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    pi = np.array([0.6, 0.4])
    assert ProbabilityEngine.transition_probability(tm, 0, 1) == pytest.approx(0.2)
    joint = ProbabilityEngine.joint_probability(tm, pi)
    assert joint.shape == (2, 2)
    assert np.isclose(joint.sum(), 1.0)
    cond = ProbabilityEngine.conditional_probability(tm)
    assert np.allclose(cond.sum(axis=1), 1.0)
    fc = ProbabilityEngine.forecast_probability(pi, tm, steps=3)
    assert fc.shape == (3, 2)
    assert ProbabilityEngine.state_probability(pi, 0) == pytest.approx(0.6)
    assert ProbabilityEngine.state_probability(np.vstack([pi, pi])).shape == (2, 2)
    bundle = ProbabilityEngine.bundle(np.vstack([pi, pi]), tm, forecast_steps=2)
    assert bundle.forecast_probabilities is not None
    assert "state_probabilities" in bundle.to_dict()
    # zero-sum normalize path
    assert np.isclose(ProbabilityEngine.normalize_rows(np.zeros(3)).sum(), 1.0)
    assert np.allclose(ProbabilityEngine.normalize_rows(np.zeros((2, 2))).sum(axis=1), 1.0)


@pytest.mark.unit
def test_persistence_engine() -> None:
    states = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    tm = np.eye(3) * 0.7 + 0.1
    tm = ProbabilityEngine.normalize_rows(tm)
    report = PersistenceEngine.analyze(states, tm, rolling_window=3)
    assert 0 in report.state_durations
    assert report.expected_duration[0] > 1.0
    assert len(report.rolling_persistence) == len(states)
    assert report.to_dict()["average_duration"]
    assert PersistenceEngine.run_lengths(np.array([])) == {}


@pytest.mark.unit
def test_forecast_helpers() -> None:
    probs = np.array([[0.5, 0.5], [0.2, 0.8]])
    fc = RegimeForecast.from_probabilities(
        probs, state_names=("a", "b"), expected_duration={0: 2.0}
    )
    assert fc.steps == 2
    assert fc.one_step().shape == (2,)
    assert fc.n_step(2)[1] == pytest.approx(0.8)
    assert "step_1" in fc.confidence_intervals
    assert fc.to_dict()["most_likely_path"] == [0, 1]
    one = RegimeForecast.from_probabilities(np.array([0.3, 0.7]))
    assert one.steps == 1
    with pytest.raises(ValueError):
        one.n_step(2)
    with pytest.raises(ValueError):
        fc.n_step(99)


@pytest.mark.unit
def test_mock_model_fit_predict_forecast_evaluate(tmp_path: Path) -> None:
    frame = _ohlcv(200)
    model = MockRegimeModel(n_states=3, window=10, random_seed=1)
    with pytest.raises(ValidationError):
        model.predict(frame)
    model.fit(frame, feature_columns=["feat_a"])
    assert model.is_fitted
    ids = model.predict(frame)
    proba = model.predict_proba(frame)
    assert ids.shape == (frame.height,)
    assert proba.shape == (frame.height, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    tm = model.transition_matrix()
    assert tm.shape == (3, 3)
    states = model.state_sequence(frame, ["feat_a"])
    assert len(states) == frame.height
    assert states[0].features_used == ("feat_a",)
    fc = model.forecast(frame, steps=5)
    assert fc.steps == 5
    result = model.detect(frame, ["feat_a"], forecast_steps=3)
    assert result.model_name == "mock_regime"
    assert result.persistence is not None
    assert result.to_frame().height == frame.height
    assert result.to_dict()["model_version"] == "1.0.0"

    report = model.evaluate(frame, true_states=ids, feature_columns=["feat_a"])
    assert report.prediction_accuracy == pytest.approx(1.0)
    assert report.n_states == 3
    unsupervised = model.evaluate(frame)
    assert unsupervised.n_samples == frame.height
    assert "prediction_accuracy" in report.to_dict()

    path = tmp_path / "mock.json"
    model.save(path)
    loaded = MockRegimeModel.load(path)
    assert loaded.is_fitted
    assert np.array_equal(loaded.predict(frame), ids)
    assert model.get_params()["n_states"] == 3


@pytest.mark.unit
def test_mock_model_short_series_and_missing_primary() -> None:
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    model = MockRegimeModel(n_states=2, window=2, primary_column="missing")
    model.fit(frame)
    assert model.predict(frame).shape == (5,)
    emptyish = pl.DataFrame({"sym": ["a", "b"]})
    with pytest.raises(ValidationError):
        MockRegimeModel(n_states=2).fit(emptyish)


@pytest.mark.unit
def test_serializer_and_trainer(tmp_path: Path) -> None:
    frame = _ohlcv(120)
    trainer = RegimeTrainer(RegimeSettings.default())
    model = trainer.train(
        frame,
        model_name="mock_regime",
        feature_columns=["feat_a"],
        artifact_path=tmp_path / "art.json",
        n_states=2,
        window=8,
    )
    assert model.meta.n_states == 2
    ser = RegimeSerializer()
    path = ser.save(model, tmp_path / "m.json")
    loaded = ser.load(path, model_cls=MockRegimeModel)
    assert loaded.is_fitted

    disabled = RegimeSettings.model_validate(
        {**RegimeSettings.default().model_dump(), "enabled": False}
    )
    with pytest.raises(ConfigurationError):
        RegimeTrainer(disabled).train(frame)


@pytest.mark.unit
def test_detector_facade(tmp_path: Path) -> None:
    frame = _ohlcv(140)
    settings = RegimeSettings.model_validate(
        {
            **RegimeSettings.default().model_dump(),
            "store_dir": str(tmp_path / "store"),
            "duckdb_path": str(tmp_path / "store" / "r.duckdb"),
            "output_dir": str(tmp_path / "out"),
            "visualization": {"enabled": True, "max_points": 100},
        }
    )
    detector = RegimeDetector(settings)
    assert "mock_regime" in detector.available_models()
    assert detector.describe_model("mock_regime")["name"] == "mock_regime"
    result = detector.detect(
        frame,
        model_name="mock_regime",
        feature_columns=["feat_a"],
        persist=True,
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1h",
        write_charts=True,
        forecast_steps=4,
    )
    assert result.forecast is not None
    charts = list((tmp_path / "out" / "charts").glob("*.svg"))
    assert len(charts) == 4
    store_stats = detector.store.stats()
    assert store_stats["file_count"] >= 2

    model = detector.fit(frame, model_name="mock_regime")
    ev = detector.evaluate(model, frame, true_states=model.predict(frame))
    assert ev.prediction_accuracy == pytest.approx(1.0)
    art = detector.save(model, tmp_path / "det.json")
    loaded = detector.load(art, model_name="mock_regime")
    assert loaded.is_fitted

    # fit=False path with unfitted registry create -> auto fit
    result2 = detector.detect(frame, fit=False, write_charts=False, persist=False)
    assert len(result2.state_ids) == frame.height


@pytest.mark.unit
def test_store_read_and_charts(tmp_path: Path) -> None:
    frame = _ohlcv(100)
    model = MockRegimeModel(window=5).fit(frame)
    result = model.detect(frame, forecast_steps=2)
    store = RegimeStore(
        tmp_path / "reg",
        duckdb_path=tmp_path / "reg" / "r.duckdb",
        register_duckdb=True,
    )
    paths = store.write_result(result, exchange="BINANCE", symbol="ETHUSDT", timeframe="1h")
    assert paths["states"].exists()
    assert paths["forecast"].exists()
    df = store.read_states(
        exchange="binance",
        symbol="ETHUSDT",
        timeframe="1h",
        model_name="mock_regime",
        version="1.0.0",
    )
    assert df.height == frame.height
    empty = store.read_states(
        exchange="binance", symbol="NOPE", timeframe="1h", model_name="mock_regime"
    )
    assert empty.is_empty()

    plot_timeline(result, tmp_path / "t.svg")
    plot_transitions(result, tmp_path / "tr.svg")
    plot_persistence(result, tmp_path / "p.svg")
    plot_probabilities(result, tmp_path / "pr.svg")
    # empty persistence path
    result.persistence = None
    plot_persistence(result, tmp_path / "p2.svg")


@pytest.mark.unit
def test_evaluator_edge_cases() -> None:
    ev = RegimeEvaluator()
    pred = np.array([0], dtype=np.int64)
    proba = np.array([[1.0, 0.0]])
    tm = np.eye(2)
    report = ev.evaluate(
        predicted=pred, probabilities=proba, transition_matrix=tm, true_states=pred
    )
    assert np.isnan(report.transition_accuracy)
    empty = ev.evaluate(
        predicted=np.array([], dtype=np.int64),
        probabilities=np.zeros((0, 2)),
        transition_matrix=tm,
    )
    assert empty.n_samples == 0


@pytest.mark.unit
def test_register_decorator_and_meta() -> None:
    @register_regime_model
    class Tiny(MockRegimeModel):
        meta = RegimeModelMeta(
            name="tiny_test_regime",
            version="0.0.1",
            description="ephemeral",
            n_states=2,
            algorithm_family="mock",
            state_names=("a", "b"),
        )

    assert "tiny_test_regime" in get_registry().list_names()


@pytest.mark.unit
def test_coverage_edges(tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    from iqrp.app.regimes.services.serializer import _json_default

    cfg = OmegaConf.create({"enabled": True, "default_model": "mock_regime"})
    assert RegimeSettings.from_mapping(cfg).default_model == "mock_regime"
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not_a_mapping\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        RegimeSettings.from_hydra(bad)

    one = RegimeForecast.from_probabilities(np.array([0.4, 0.6]))
    assert one.n_step(1).shape == (2,)
    assert ProbabilityEngine.joint_probability(np.eye(2), np.zeros(2)).shape == (2, 2)
    bundle0 = ProbabilityEngine.bundle(np.ones((3, 2)) / 2, np.eye(2), forecast_steps=0)
    assert bundle0.forecast_probabilities is None

    assert isinstance(_json_default(np.array([1.0])), list)
    assert _json_default(np.float64(1.5)) == 1.5
    with pytest.raises(TypeError):
        _json_default(object())

    state = RegimeState.from_dict(
        {
            "state_id": 0,
            "state_name": "a",
            "probability": 1.0,
            "confidence": 1.0,
            "persistence": 0.5,
            "start_time": datetime(2024, 1, 1, tzinfo=UTC),
            "timestamp": None,
        }
    )
    assert state.start_time is not None

    store = RegimeStore(tmp_path / "empty_store", register_duckdb=True)
    store._register_duckdb(tmp_path / "no_partition")

    frame = _ohlcv(80)
    model = MockRegimeModel(window=5).fit(frame)
    result = model.detect(frame)
    result.state_probabilities = np.array([0.3, 0.7])
    plot_probabilities(result, tmp_path / "bad_proba.svg")

    local = RegimeModelRegistry()
    local.register(MockRegimeModel)
    local.clear()
    assert local.list_names() == []

    bare = MockRegimeModel()
    bare._fitted = True
    bare._transition_matrix = None
    with pytest.raises(ValidationError):
        bare.transition_matrix()
