"""Gap-filling tests to push Markov engine coverage above 98%."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from iqrp.app.core.exceptions import ConfigurationError, ValidationError
from iqrp.app.regimes.markov.config import MarkovSettings
from iqrp.app.regimes.markov.evaluator import _avg_cross_entropy, next_state_accuracy
from iqrp.app.regimes.markov.forecast import _as_distribution
from iqrp.app.regimes.markov.model import MarkovChainModel, MarkovRegimeModel, _resolve_names
from iqrp.app.regimes.markov.persistence import PersistenceAnalyzer, _run_lengths
from iqrp.app.regimes.markov.state_mapper import LabelStateMapper
from iqrp.app.regimes.markov.stationary import estimate_period, is_irreducible
from iqrp.app.regimes.markov.transition import TransitionMatrix
from iqrp.app.regimes.markov.visualization import plot_forecast_probabilities


@pytest.mark.unit
def test_config_root_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        MarkovSettings.from_hydra(bad)


@pytest.mark.unit
def test_resolve_names_and_mapper_edges() -> None:
    assert _resolve_names(3, ("a",)) == ("a", "state_1", "state_2")
    m = LabelStateMapper(n_states=2)
    assert m.n_states == 2
    m2 = LabelStateMapper(label_to_id={"x": 0})
    m2.fit(["x", "x"])
    assert m2.transform(["x"])[0] == 0
    m3 = LabelStateMapper(custom_mapper=lambda z: 1)
    m3.fit(["a"])
    assert m3.transform(["z"])[0] == 1
    assert (
        LabelStateMapper.from_dict({"n_states": 2, "fixed": {}, "learned": {"q": 0}}).map_one("q")
        == 0
    )


@pytest.mark.unit
def test_evaluator_transition_matrix_ce_and_short() -> None:
    tm = np.array([[0.8, 0.2], [0.3, 0.7]])
    states = np.array([0, 1, 0, 1])
    ce = _avg_cross_entropy(tm, states)
    assert np.isfinite(ce)
    assert next_state_accuracy(np.array([0]), tm) == 1.0
    assert np.isnan(_avg_cross_entropy(np.array([0.5, 0.5]), np.array([0])))


@pytest.mark.unit
def test_forecast_oob_state_id() -> None:
    d = _as_distribution(np.array([9]), n_states=3)
    assert d.sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_model_short_series_and_columns() -> None:
    model = MarkovChainModel(n_states=2)
    model.fit(np.array([0]))
    report = model.evaluate(np.array([0]))
    assert "aic" in report["metrics"]
    frame = pl.DataFrame({"regime_state": [0, 1, 0, 1]})
    model2 = MarkovChainModel(n_states=2)
    model2.fit(frame)
    assert model2.predict(frame).shape == (4,)
    proba = model2._state_probabilities(np.array([0, 99]))
    assert proba[1].sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_diagnostics_no_train_states() -> None:
    model = MarkovChainModel(n_states=2)
    model.fit(np.array([0, 1, 0]))
    model._train_states = None
    with pytest.raises(ValidationError):
        model.diagnostics()
    empty = PersistenceAnalyzer().analyze(np.array([], dtype=np.int64), n_states=0)
    assert empty["mean_run_length"] == 0.0 or empty["state_occupancy"]["counts"] == []


@pytest.mark.unit
def test_partial_fit_before_fitted_and_weights_window() -> None:
    settings = MarkovSettings.from_mapping(
        {
            **MarkovSettings.default().model_dump(),
            "n_states": 2,
            "estimation": {
                "method": "bayesian",
                "laplace_alpha": 1.0,
                "dirichlet_alpha": 1.0,
                "forgetting_factor": 1.0,
                "window_size": 5,
                "min_count_warning": 1,
            },
        }
    )
    model = MarkovChainModel(n_states=2, settings=settings)
    model.partial_fit(np.array([0, 1, 0, 1, 0, 1, 0, 1]))
    assert model.is_fitted
    model.fit(np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]), weights=np.ones(10))


@pytest.mark.unit
def test_transition_sparse_threshold_and_forgetting() -> None:
    tm = TransitionMatrix(2, laplace_alpha=0.0, sparse_threshold=0.5)
    tm.update_sequence([0, 0, 0, 1])
    sp = tm.sparse_probability_matrix()
    assert sp.shape == (2, 2)
    tm2 = TransitionMatrix(2)
    tm2.update_sequence([0, 1, 0], forgetting_factor=0.5)
    assert tm2.n_transitions >= 1


@pytest.mark.unit
def test_stationary_empty_and_period() -> None:
    assert estimate_period(np.zeros((0, 0))) == 1
    p = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert is_irreducible(p) is True
    assert estimate_period(p) >= 1
    bad = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert is_irreducible(bad) is False


@pytest.mark.unit
def test_persistence_empty_runs() -> None:
    assert _run_lengths(np.array([], dtype=np.int64)) == {}
    # Identity has P_ii=1 => huge expected duration
    assert PersistenceAnalyzer().expected_duration(np.eye(2))[0] > 1e6
    soft = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert PersistenceAnalyzer().expected_duration(soft)[0] == pytest.approx(2.0)


@pytest.mark.unit
def test_regime_model_import_state() -> None:
    reg = MarkovRegimeModel(n_states=2)
    frame = pl.DataFrame({"state_id": [0, 0, 1, 1, 0]})
    reg.fit(frame)
    payload = reg.export_state()
    reg2 = MarkovRegimeModel(n_states=2)
    reg2.import_state(payload)
    assert reg2.is_fitted
    assert reg2.predict(frame).shape == (5,)


@pytest.mark.unit
def test_viz_1d_forecast(tmp_path: Path) -> None:
    plot_forecast_probabilities(np.array([0.2, 0.8]), tmp_path / "f.svg")
    assert (tmp_path / "f.svg").exists()


@pytest.mark.unit
def test_callable_mapper_on_model() -> None:
    model = MarkovChainModel(n_states=2, state_mapper=lambda x: int(x) % 2)
    model.fit(np.array([0, 1, 2, 3]))
    assert model.predict(np.array([4, 5])).tolist() == [0, 1]


@pytest.mark.unit
def test_omega_list_config_invalid() -> None:
    with pytest.raises(ConfigurationError):
        MarkovSettings.from_mapping(OmegaConf.create([1, 2]))
