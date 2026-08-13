"""Extra coverage for remaining state-space branches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from iqrp.app.core.exceptions import ConfigurationError
from iqrp.app.state_space import StateSpaceSettings, get_registry
from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.base.forecast_result import ForecastResult
from iqrp.app.state_space.base.observation_model import DiagonalGaussianObservationModel
from iqrp.app.state_space.base.registry import state_space_model_factory
from iqrp.app.state_space.evaluation.diagnostics import probability_calibration
from iqrp.app.state_space.evaluation.metrics import sequence_cross_entropy
from iqrp.app.state_space.forecasting.uncertainty import forecast_uncertainty
from iqrp.app.state_space.models.mock import MockDiscreteStateSpaceModel
from iqrp.app.state_space.smoothing.fixed_interval import FixedIntervalSmoother
from iqrp.app.state_space.storage.serializer import StateSpaceSerializer, _json_default
from iqrp.app.state_space.storage.state_store import StateStore
from iqrp.app.state_space.visualization import (
    plot_forecast_uncertainty,
    plot_persistence_distribution,
    plot_probability_heatmap,
    plot_transition_graph,
)


@pytest.mark.unit
def test_observation_base_methods() -> None:
    om = DiagonalGaussianObservationModel([[0.0], [1.0]], [[1.0], [1.0]])
    em = om.emission_matrix(np.array([0.0, 1.0, 0.5]))
    assert em.shape == (3, 2)
    assert om.log_emission([0.0], 0) > om.log_emission([0.0], 1)
    assert om.soft_responsibilities(np.array([[0.0], [1.0]])).shape == (2, 2)
    assert om.predictive_density([0.0], [0.5, 0.5]) > 0
    assert om.expected_observation(0).shape == (1,)


@pytest.mark.unit
def test_config_omega_and_default_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = OmegaConf.create({"enabled": True, "forecasting": {"default_horizon": 3}})
    s = StateSpaceSettings.from_mapping(cfg)
    assert s.forecasting.default_horizon == 3
    with pytest.raises(ConfigurationError):
        StateSpaceSettings.from_mapping(OmegaConf.create([1, 2]))  # type: ignore[arg-type]

    missing = tmp_path / "nope.yaml"
    monkeypatch.setattr(
        "iqrp.app.state_space.config._default_config_path",
        lambda: missing,
    )
    assert StateSpaceSettings.default().enabled is True


@pytest.mark.unit
def test_registry_helpers() -> None:
    reg = get_registry()
    assert reg.all_meta()
    factory = state_space_model_factory("mock_discrete_ssm")
    assert factory is MockDiscreteStateSpaceModel
    assert reg.describe("mock_discrete_ssm").name == "mock_discrete_ssm"


@pytest.mark.unit
def test_forecast_most_likely_without_steps() -> None:
    fc = ForecastResult(
        horizon=3,
        expected_state=1,
        probability_distribution=np.array([0.2, 0.8]),
        confidence_interval=(0.8, 0.8),
        expected_duration={0: 1.0, 1: 2.0},
        step_distributions=None,
    )
    assert fc.most_likely_path == [1, 1, 1]
    assert (
        FilterResult(
            filtered_states=[0],
            filtered_probabilities=np.array([1.0, 0.0]),
            log_likelihood=0.0,
            normalization_constants=[1.0],
        ).n_states
        == 2
    )


@pytest.mark.unit
def test_fixed_interval_without_filter_result() -> None:
    log_e = np.zeros((8, 2))
    p = np.array([[0.7, 0.3], [0.4, 0.6]])
    smooth = FixedIntervalSmoother().run(log_e, p)
    assert smooth.n_steps == 8


@pytest.mark.unit
def test_serializer_json_default_and_store_edges(tmp_path: Path) -> None:
    assert _json_default(np.array([1.0])) == [1.0]
    assert _json_default(np.float64(1.5)) == 1.5
    with pytest.raises(TypeError):
        _json_default(object())

    store = StateStore(
        root=tmp_path / "s", duckdb_path=tmp_path / "d.duckdb", register_duckdb=False
    )
    assert store.read_states(exchange="x", symbol="y", timeframe="1h", model_name="m").height == 0
    filt = FilterResult(
        filtered_states=[0],
        filtered_probabilities=[[1.0]],
        log_likelihood=0.0,
        normalization_constants=[1.0],
    )
    store.write_filter_result(filt, model_name="m", version="1")
    assert store.stats()["file_count"] >= 1


@pytest.mark.unit
def test_viz_disabled_and_1d(tmp_path: Path) -> None:
    settings = StateSpaceSettings.from_mapping(
        {
            **StateSpaceSettings.default().model_dump(),
            "visualization": {"enabled": False, "max_points": 10},
        }
    )
    plot_probability_heatmap(np.array([0.2, 0.8]), tmp_path / "a.svg", settings)
    plot_transition_graph(np.eye(2), tmp_path / "b.svg", settings)
    plot_persistence_distribution([], tmp_path / "c.svg", settings)
    plot_forecast_uncertainty(np.array([0.5, 0.5]), tmp_path / "d.svg", settings)
    unc = forecast_uncertainty(np.array([0.2, 0.8]))
    assert "entropy" in unc
    cal = probability_calibration(np.array([[0.9, 0.1]]), np.array([0]))
    assert "ece" in cal
    assert sequence_cross_entropy(np.eye(2), np.array([0, 1])) >= 0


@pytest.mark.unit
def test_mock_settings_columns_and_fixed_lag() -> None:
    settings = StateSpaceSettings.from_mapping(
        {
            **StateSpaceSettings.default().model_dump(),
            "columns": {"timestamp": "open_time", "observation_columns": ["x"]},
            "smoothing": {"algorithm": "fixed_lag", "fixed_lag": 2},
        }
    )
    frame = pl.DataFrame({"open_time": [0, 1, 2, 3, 4], "x": [0.0, 0.1, -0.1, 0.2, 0.0]})
    model = MockDiscreteStateSpaceModel(n_states=2, settings=settings)
    model.fit(frame)
    assert model.smooth(frame).metadata["algorithm"] == "fixed_lag"
    bad = pl.DataFrame({"open_time": [0, 1], "symbol": ["a", "b"]})
    model2 = MockDiscreteStateSpaceModel(n_states=2)
    from iqrp.app.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        model2.fit(bad)


@pytest.mark.unit
def test_serializer_roundtrip_arrays(tmp_path: Path) -> None:
    model = MockDiscreteStateSpaceModel(n_states=2, random_seed=0)
    model.fit(np.linspace(-1, 1, 30))
    path = StateSpaceSerializer().save(model, tmp_path / "m.json")
    assert path.with_suffix(".npz").exists()
    loaded = StateSpaceSerializer().load(path, model_cls=MockDiscreteStateSpaceModel)
    assert loaded.is_fitted
