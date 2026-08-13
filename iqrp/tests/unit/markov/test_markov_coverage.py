"""Coverage, property, and edge-case tests for Markov engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings, strategies as st
from omegaconf import OmegaConf

from iqrp.app.core.exceptions import ConfigurationError, ValidationError
from iqrp.app.regimes.markov import (
    LabelStateMapper,
    MarkovChainModel,
    MarkovSettings,
    TransitionMatrix,
)
from iqrp.app.regimes.markov.evaluator import next_state_accuracy
from iqrp.app.regimes.markov.forecast import _as_distribution
from iqrp.app.regimes.markov.serializer import _json_default
from iqrp.app.regimes.markov.stationary import estimate_period, is_ergodic, spectral_gap
from iqrp.app.regimes.markov.visualization import (
    plot_forecast_probabilities,
    plot_persistence_histogram,
    plot_stationary_distribution,
    plot_transition_graph,
    plot_transition_heatmap,
)
from iqrp.app.simulation.regimes.regime_switching import RegimeSwitchingSimulator


@pytest.mark.unit
def test_config_invalid_and_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError):
        MarkovSettings.from_mapping([1, 2, 3])  # type: ignore[arg-type]
    cfg = OmegaConf.create({"n_states": 2})
    assert MarkovSettings.from_mapping(cfg).n_states == 2
    with pytest.raises(ConfigurationError):
        MarkovSettings.from_mapping(OmegaConf.create([1]))  # type: ignore[arg-type]
    monkeypatch.setattr(
        "iqrp.app.regimes.markov.config._default_config_path",
        lambda: tmp_path / "missing.yaml",
    )
    assert MarkovSettings.default().n_states == 3


@pytest.mark.unit
def test_transition_validation_errors() -> None:
    with pytest.raises(ValidationError):
        TransitionMatrix(0)
    model = MarkovChainModel(n_states=2)
    with pytest.raises(ValidationError):
        model.transition_matrix()
    with pytest.raises(ValidationError):
        model.fit(pl.DataFrame({"open_time": [0, 1], "symbol": ["a", "b"]}))


@pytest.mark.unit
def test_mapper_custom_and_unknown() -> None:
    m = LabelStateMapper(custom_mapper=lambda x: 0 if x == "low" else 1)
    assert m.transform(["low", "high"]).tolist() == [0, 1]
    m2 = LabelStateMapper(n_states=2)
    assert m2.map_one(1) == 1
    with pytest.raises(ValidationError):
        LabelStateMapper(label_to_id={"a": 0}).map_one("missing")


@pytest.mark.unit
def test_forecast_as_distribution_state_id() -> None:
    d = _as_distribution(np.array([1]), n_states=3)
    assert d.tolist() == [0.0, 1.0, 0.0]
    assert _as_distribution(np.array([0.2, 0.8])).sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_serializer_json_default() -> None:
    assert _json_default(np.array([1.0])) == [1.0]
    assert _json_default(np.float64(2.0)) == 2.0
    with pytest.raises(TypeError):
        _json_default(object())


@pytest.mark.unit
def test_online_sliding_window() -> None:
    settings = MarkovSettings.from_mapping(
        {
            **MarkovSettings.default().model_dump(),
            "estimation": {
                "method": "bayesian",
                "laplace_alpha": 1.0,
                "dirichlet_alpha": 1.0,
                "forgetting_factor": 0.9,
                "window_size": 50,
                "min_count_warning": 2,
            },
            "online": {"update_frequency": 1, "adaptive": True},
        }
    )
    model = MarkovChainModel(n_states=2, settings=settings)
    s = np.array([0, 0, 1, 1, 0, 0, 1] * 20, dtype=np.int64)
    model.fit(s)
    model.partial_fit(np.array([0, 1, 0, 1, 0], dtype=np.int64))
    assert model.is_fitted


@pytest.mark.unit
def test_weighted_frame_and_numpy() -> None:
    s = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    frame = pl.DataFrame({"state_id": s.tolist(), "w": [1.0] * len(s)})
    settings = MarkovSettings.from_mapping(
        {
            **MarkovSettings.default().model_dump(),
            "n_states": 2,
            "estimation": {
                "method": "weighted",
                "laplace_alpha": 0.5,
                "dirichlet_alpha": 1.0,
                "forgetting_factor": 1.0,
                "window_size": 0,
                "min_count_warning": 1,
            },
            "columns": {"timestamp": "open_time", "state_column": "state_id", "weight_column": "w"},
        }
    )
    model = MarkovChainModel(n_states=2, settings=settings)
    model.fit(frame)
    model.fit(s, weights=np.ones(len(s)))
    assert next_state_accuracy(s, model.transition_matrix()) >= 0.0


@pytest.mark.unit
def test_stationary_period_helpers() -> None:
    p = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert estimate_period(p) >= 1
    assert spectral_gap(np.eye(1)) == 0.0
    assert is_ergodic(RegimeSwitchingSimulator.mixed_transition(3, 0.8)) is True


@pytest.mark.unit
def test_viz_disabled_branches(tmp_path: Path) -> None:
    settings = MarkovSettings.from_mapping(
        {
            **MarkovSettings.default().model_dump(),
            "visualization": {"enabled": False, "max_points": 5},
        }
    )
    p = np.eye(2)
    plot_transition_heatmap(p, tmp_path / "a.svg", settings)
    plot_transition_graph(p, tmp_path / "b.svg", settings)
    plot_persistence_histogram({}, tmp_path / "c.svg", settings)
    plot_forecast_probabilities(np.array([0.5, 0.5]), tmp_path / "d.svg", settings)
    plot_stationary_distribution([0.5, 0.5], tmp_path / "e.svg", settings)


@given(
    persistence=st.floats(0.7, 0.98),
    n=st.integers(80, 200),
)
@settings(max_examples=8, deadline=None)
def test_property_rows_stochastic(persistence: float, n: int) -> None:
    k = 3
    true_p = RegimeSwitchingSimulator.mixed_transition(k, persistence)
    path = RegimeSwitchingSimulator(np.random.default_rng(0)).simulate(
        n,
        transition_matrix=true_p,
        state_names=tuple(f"s{i}" for i in range(k)),
        drifts=[0.0] * k,
        volatilities=[0.2] * k,
    )
    model = MarkovChainModel(n_states=k)
    model.fit(path.state_ids)
    est = model.transition_matrix()
    assert np.allclose(est.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(est >= -1e-12)


@pytest.mark.unit
def test_diagnostics_without_observations() -> None:
    model = MarkovChainModel(n_states=2)
    model.fit(np.array([0, 0, 1, 1, 0]))
    assert model.diagnostics()["state_frequencies"]
    assert model.persistence_report()["mean_run_length"] >= 0
