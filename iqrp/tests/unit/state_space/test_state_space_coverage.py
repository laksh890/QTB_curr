"""Coverage / edge-case tests for the State Space Framework."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.core.exceptions import ConfigurationError, ValidationError
from iqrp.app.state_space import (
    BackwardFilter,
    DiagonalGaussianObservationModel,
    MatrixTransitionModel,
    StateSpaceSettings,
    get_registry,
)
from iqrp.app.state_space.base.probabilities import forward_probabilities
from iqrp.app.state_space.base.registry import StateSpaceModelRegistry, register_state_space_model
from iqrp.app.state_space.base.state_space_model import StateSpaceModel, StateSpaceModelMeta
from iqrp.app.state_space.evaluation.diagnostics import (
    likelihood_convergence_checks,
    residual_diagnostics,
)
from iqrp.app.state_space.evaluation.metrics import (
    persistence_stability,
    sequence_cross_entropy,
    state_stability,
    transition_accuracy,
)
from iqrp.app.state_space.forecasting.uncertainty import probability_interval
from iqrp.app.state_space.models.mock import MockDiscreteStateSpaceModel
from iqrp.app.state_space.visualization import (
    plot_forecast_uncertainty,
    plot_persistence_distribution,
    plot_probability_heatmap,
    plot_state_timeline,
    plot_transition_graph,
)


@pytest.mark.unit
def test_settings_invalid_mapping() -> None:
    with pytest.raises(ConfigurationError):
        StateSpaceSettings.from_mapping([1, 2, 3])  # type: ignore[arg-type]


@pytest.mark.unit
def test_transition_shape_error() -> None:
    with pytest.raises(ValidationError):
        MatrixTransitionModel([[0.5, 0.5]])


@pytest.mark.unit
def test_observation_shape_error() -> None:
    with pytest.raises(ValidationError):
        DiagonalGaussianObservationModel(means=[[0.0]], variances=[[0.1], [0.2]])


@pytest.mark.unit
def test_model_not_fitted_errors() -> None:
    model = MockDiscreteStateSpaceModel(n_states=2)
    with pytest.raises(ValidationError):
        model.filter(np.zeros((5, 1)))


@pytest.mark.unit
def test_registry_errors() -> None:
    reg = StateSpaceModelRegistry()

    class Bad:
        pass

    with pytest.raises(ConfigurationError):
        reg.register(Bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        get_registry().get_class("does_not_exist")


@pytest.mark.unit
def test_forward_probabilities_shape_error() -> None:
    with pytest.raises(ValueError):
        forward_probabilities(np.array([1.0, 2.0]), np.eye(2))


@pytest.mark.unit
def test_backward_filter_and_chunk_paths() -> None:
    log_e = np.zeros((5, 2))
    p = np.array([[0.6, 0.4], [0.3, 0.7]])
    bf = BackwardFilter()
    result = bf.run(log_e, p)
    assert result.log_messages is not None
    beta = bf.backward_messages(log_e, p, scales=result.normalization_constants)
    assert beta.shape == (5, 2)


@pytest.mark.unit
def test_metric_edge_cases() -> None:
    assert state_stability(np.array([1])) == 1.0
    assert persistence_stability(np.array([])) == 0.0
    assert transition_accuracy(np.array([0]), np.array([0])) == 1.0
    assert np.isnan(sequence_cross_entropy(np.array([0.5, 0.5]), np.array([0])))
    assert residual_diagnostics(None, None)["available"] is False
    assert likelihood_convergence_checks(None)["available"] is False
    assert probability_interval([0.1, 0.9])[1] >= probability_interval([0.1, 0.9])[0]


@pytest.mark.unit
def test_visualization_svg(tmp_path: Path) -> None:
    settings = StateSpaceSettings.default()
    plot_state_timeline(np.array([0, 1, 0, 1]), tmp_path / "tl.svg", settings)
    plot_probability_heatmap(np.array([[0.2, 0.8], [0.5, 0.5]]), tmp_path / "hm.svg", settings)
    plot_transition_graph(np.array([[0.9, 0.1], [0.2, 0.8]]), tmp_path / "tg.svg", settings)
    plot_persistence_distribution([1, 2, 2, 5], tmp_path / "pd.svg", settings)
    plot_forecast_uncertainty(np.array([[0.5, 0.5], [0.2, 0.8]]), tmp_path / "fu.svg", settings)
    disabled = StateSpaceSettings.from_mapping(
        {**settings.model_dump(), "visualization": {"enabled": False, "max_points": 10}}
    )
    plot_state_timeline(np.array([0]), tmp_path / "off.svg", disabled)
    assert (tmp_path / "tl.svg").read_text().startswith("<svg")


@pytest.mark.unit
def test_mock_numpy_and_close_fallback() -> None:
    y = np.concatenate([np.full(20, -1.0), np.full(20, 1.0)])
    model = MockDiscreteStateSpaceModel(n_states=2, random_seed=1)
    model.fit(y)
    assert model.filter(y).n_states == 2
    # Polars with only close
    frame = pl.DataFrame({"open_time": list(range(40)), "close": y})
    model2 = MockDiscreteStateSpaceModel(n_states=2)
    model2.fit(frame)
    assert model2.predict(frame).shape == (40,)


@pytest.mark.unit
def test_filter_result_to_frame_and_hard_states() -> None:
    from iqrp.app.state_space.filtering.base_filter import BaseFilter

    class _F(BaseFilter):
        def run(self, log_emissions, transition, *, initial=None):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    f = _F()
    assert f.hard_states(np.array([0.1, 0.9]))[0] == 1
    from iqrp.app.state_space.base.filter_result import FilterResult

    fr = FilterResult(
        filtered_states=[0, 1],
        filtered_probabilities=[[1.0, 0.0], [0.0, 1.0]],
        log_likelihood=0.0,
        normalization_constants=[1.0, 1.0],
        log_messages=[[0.0, -20.0], [-20.0, 0.0]],
    )
    assert "proba_0" in fr.to_frame(timestamps=[0, 1]).columns
    assert FilterResult.from_dict(fr.to_dict()).log_messages is not None


@pytest.mark.unit
def test_custom_register_decorator() -> None:
    @register_state_space_model
    class Tiny(StateSpaceModel):
        meta = StateSpaceModelMeta(
            name="tiny_test_ssm",
            version="0.0.1",
            description="tiny",
            n_states=1,
            algorithm_family="mock",
        )

        def fit(self, observations, *, observation_columns=None):  # type: ignore[no-untyped-def]
            self._fitted = True
            return self

        def filter(self, observations, *, observation_columns=None):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def smooth(self, observations, *, observation_columns=None, lag=None):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def predict(self, observations, *, observation_columns=None):  # type: ignore[no-untyped-def]
            return np.zeros(1, dtype=np.int64)

        def predict_proba(self, observations, *, observation_columns=None):  # type: ignore[no-untyped-def]
            return np.ones((1, 1))

        def forecast(self, observations, *, horizon=None, observation_columns=None):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def sample(self, n_steps, *, initial_state=None, rng=None):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def log_likelihood(self, observations, *, observation_columns=None):  # type: ignore[no-untyped-def]
            return 0.0

        def _algorithm_state(self):  # type: ignore[no-untyped-def]
            return {}

        def _load_algorithm_state(self, state):  # type: ignore[no-untyped-def]
            return None

    assert "tiny_test_ssm" in get_registry().list_names()
