"""Unit tests for the Markov Chain Engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.markov import (
    LabelStateMapper,
    MarkovChainModel,
    MarkovDiagnostics,
    MarkovEvaluator,
    MarkovForecaster,
    MarkovRegimeModel,
    MarkovSettings,
    MarkovTrainer,
    PersistenceAnalyzer,
    StationaryAnalyzer,
    TransitionEstimator,
    TransitionMatrix,
)
from iqrp.app.regimes.markov.visualization import (
    plot_forecast_probabilities,
    plot_occupancy_timeline,
    plot_persistence_histogram,
    plot_stationary_distribution,
    plot_transition_graph,
    plot_transition_heatmap,
)
from iqrp.app.simulation.regimes.regime_switching import RegimeSwitchingSimulator


def _states(n: int = 200, k: int = 3, seed: int = 0, persistence: float = 0.9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = RegimeSwitchingSimulator.mixed_transition(k, persistence)
    s = [0]
    for _ in range(n - 1):
        s.append(int(rng.choice(k, p=p[s[-1]])))
    return np.asarray(s, dtype=np.int64)


@pytest.mark.unit
def test_settings_hydra() -> None:
    s = MarkovSettings.from_hydra(overrides=["n_states=4", "estimation.method=mle"])
    assert s.n_states == 4
    assert s.estimation.method == "mle"
    assert MarkovSettings.default().enabled is True


@pytest.mark.unit
def test_transition_matrix_ops() -> None:
    tm = TransitionMatrix(3, laplace_alpha=1.0)
    tm.update_sequence([0, 0, 1, 1, 2, 0])
    tm.update_pair(0, 1, weight=2.0)
    assert tm.n_transitions >= 5
    p = tm.probability_matrix()
    assert p.shape == (3, 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert tm.validate()
    sp = tm.sparse_probability_matrix()
    assert sp.shape == (3, 3)
    p2 = tm.apply_window([0, 1, 0, 1, 0, 1], window_size=4)
    assert np.allclose(p2.sum(axis=1), 1.0)
    assert TransitionMatrix.from_dict(tm.to_dict()).n_states == 3


@pytest.mark.unit
def test_estimators() -> None:
    s = _states(100, 3, seed=1)
    for method in ("mle", "bayesian", "frequency", "weighted"):
        est = TransitionEstimator(3, method=method, forgetting_factor=0.95)
        w = np.ones(len(s)) if method == "weighted" else None
        p = est.fit(s, weights=w)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-8)
        est.partial_fit(s[:20], weights=w[:20] if w is not None else None)
        assert np.isfinite(est.log_likelihood(s))
        assert TransitionEstimator.from_dict(est.to_dict()).method == method


@pytest.mark.unit
def test_stationary_persistence_forecast() -> None:
    p = RegimeSwitchingSimulator.mixed_transition(3, 0.9)
    st = StationaryAnalyzer().analyze(p)
    assert st["stationary_distribution"].sum() == pytest.approx(1.0)
    assert st["is_irreducible"] is True
    persist = PersistenceAnalyzer().analyze(_states(80, 3), p, n_states=3)
    assert "expected_duration" in persist
    pi = np.array([1.0, 0.0, 0.0])
    fc = MarkovForecaster().forecast(pi, p, horizon=5)
    assert fc.horizon == 5
    assert len(fc.most_likely_path) == 5
    assert MarkovForecaster().one_step(pi, p).sum() == pytest.approx(1.0)
    assert MarkovForecaster().n_step(pi, p, 3).sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_markov_chain_model_api(tmp_path: Path) -> None:
    s = _states(150, 3, seed=2)
    frame = pl.DataFrame({"open_time": list(range(len(s))), "state_id": s.tolist()})
    model = MarkovChainModel(n_states=3, random_seed=2)
    model.fit(frame)
    assert model.is_fitted
    assert model.transition_matrix().shape == (3, 3)
    assert model.stationary_distribution().sum() == pytest.approx(1.0)
    assert model.expected_duration()
    assert model.predict(frame).shape == (len(s),)
    assert model.predict_proba(frame).shape == (len(s), 3)
    assert model.state_probabilities(frame).shape[0] == len(s)
    filt = model.filter(frame)
    assert filt.n_steps == len(s)
    smooth = model.smooth(frame)
    assert smooth.n_steps == len(s)
    fc = model.forecast(frame, horizon=4)
    assert fc.horizon == 4
    states, obs = model.sample(40, initial_state=0)
    assert states.shape == (40,) and obs.shape[0] == 40
    assert np.isfinite(model.log_likelihood(frame))
    report = model.evaluate(frame, true_states=s)
    assert "aic" in report["metrics"]
    diag = model.diagnostics(frame)
    assert "rare_states" in diag
    assert model.persistence_report(frame)["state_occupancy"]
    assert model.stationary_analysis()["is_ergodic"] in (True, False)

    model.partial_fit(s[:30])
    path = model.save(tmp_path / "mc.json")
    loaded = MarkovChainModel.load(path)
    assert loaded.is_fitted
    assert np.allclose(loaded.transition_matrix(), model.transition_matrix(), atol=1e-8)


@pytest.mark.unit
def test_regime_adapter_and_mapper() -> None:
    s = _states(60, 2, seed=3)
    frame = pl.DataFrame({"state_id": s.tolist(), "close": np.linspace(1, 2, len(s))})
    reg = MarkovRegimeModel(n_states=2)
    reg.fit(frame, feature_columns=["state_id"])
    assert reg.predict(frame, feature_columns=["state_id"]).shape == (len(s),)
    assert reg.predict_proba(frame, feature_columns=["state_id"]).shape[1] == 2
    fc = reg.forecast(frame, steps=3)
    assert fc.steps == 3

    mapper = LabelStateMapper(label_to_id={"a": 0, "b": 1})
    assert mapper.transform(["a", "b", "a"]).tolist() == [0, 1, 0]
    mapper2 = LabelStateMapper()
    mapper2.fit(["x", "y", "x"])
    assert mapper2.n_states == 2
    assert LabelStateMapper.from_dict(mapper2.to_dict()).transform(["x"])[0] == 0


@pytest.mark.unit
def test_trainer_evaluator_diagnostics() -> None:
    s = _states(100, 3, seed=4)
    model = MarkovChainModel(n_states=3)
    trainer = MarkovTrainer()
    stats = trainer.train(model, s)
    assert stats["evaluation"]["metrics"]["log_likelihood"]
    partial = trainer.partial_train(model, s[:25])
    assert partial["mode"] == "partial_fit"
    ev = MarkovEvaluator().evaluate(
        true_states=s,
        predicted_states=s,
        probabilities=np.eye(3)[s],
        transition=model.transition_matrix(),
        log_likelihood=-10.0,
        n_params=6,
        forecast_true=s[1:],
        forecast_pred=s[1:],
    )
    assert ev["metrics"]["forecast_accuracy"] == pytest.approx(1.0)
    diag = MarkovDiagnostics().generate(
        states=s,
        transition=model.transition_matrix(),
        counts=model.estimator.matrix.count_matrix(),
        state_names=model.state_names,
    )
    assert diag["mean_transition_entropy"] >= 0


@pytest.mark.unit
def test_visualization(tmp_path: Path) -> None:
    p = RegimeSwitchingSimulator.mixed_transition(3, 0.85)
    s = _states(40, 3, seed=5)
    settings = MarkovSettings.default()
    plot_transition_heatmap(p, tmp_path / "hm.svg", settings, state_names=("a", "b", "c"))
    plot_transition_graph(p, tmp_path / "tg.svg", settings)
    plot_occupancy_timeline(s, tmp_path / "tl.svg", settings)
    plot_persistence_histogram([1, 2, 2, 5, 1], tmp_path / "ph.svg", settings)
    plot_forecast_probabilities(np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]]), tmp_path / "fc.svg")
    plot_stationary_distribution(np.array([0.2, 0.5, 0.3]), tmp_path / "st.svg", settings)
    disabled = MarkovSettings.from_mapping(
        {**settings.model_dump(), "visualization": {"enabled": False, "max_points": 10}}
    )
    plot_occupancy_timeline(s, tmp_path / "off.svg", disabled)
    assert (tmp_path / "hm.svg").read_text().startswith("<svg")
