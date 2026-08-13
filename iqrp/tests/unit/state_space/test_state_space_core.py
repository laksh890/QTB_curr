"""Unit tests for the State Space Framework core contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.state_space import (
    DiagonalGaussianObservationModel,
    EvaluationMetrics,
    FilterResult,
    FixedIntervalSmoother,
    FixedLagSmoother,
    ForecastResult,
    ForwardFilter,
    LatentState,
    MatrixTransitionModel,
    MultiStepForecaster,
    Observation,
    StateSpaceDiagnostics,
    StateSpaceSettings,
    StateStore,
    get_registry,
)
from iqrp.app.state_space.base.probabilities import (
    backward_probabilities,
    forecast_distribution,
    forward_probabilities,
    joint_probabilities,
    state_occupancy_probabilities,
    transition_probabilities,
)
from iqrp.app.state_space.models.mock import MockDiscreteStateSpaceModel


def _synthetic_frame(n: int = 80, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    # Two-regime-ish synthetic series
    x = np.concatenate([rng.normal(-1.0, 0.3, n // 2), rng.normal(1.0, 0.3, n - n // 2)])
    return pl.DataFrame(
        {
            "open_time": list(range(n)),
            "close": x,
            "feature_a": x + rng.normal(0, 0.05, n),
        }
    )


@pytest.mark.unit
def test_settings_hydra_overrides() -> None:
    s = StateSpaceSettings.from_hydra(overrides=["forecasting.default_horizon=7"])
    assert s.forecasting.default_horizon == 7
    assert StateSpaceSettings.default().enabled is True


@pytest.mark.unit
def test_registry_lists_mock() -> None:
    names = get_registry().list_names()
    assert "mock_discrete_ssm" in names
    meta = get_registry().describe("mock_discrete_ssm")
    assert meta.algorithm_family == "mock"


@pytest.mark.unit
def test_latent_state_and_observation_roundtrip() -> None:
    ls = LatentState(1, "s1", 0.7, 0.7, timestamp=3, duration=2.0, metadata={"a": 1})
    assert LatentState.from_dict(ls.to_dict()).state_id == 1
    obs = Observation(values=[1.0, 2.0], timestamp=0, mask=[True, False])
    assert Observation.from_dict(obs.to_dict()).dim == 2
    assert Observation.from_frame_row([0.5], timestamp=1).values[0] == 0.5


@pytest.mark.unit
def test_transition_and_observation_models() -> None:
    tm = MatrixTransitionModel([[0.8, 0.2], [0.3, 0.7]])
    assert tm.validate()
    assert tm.transition_probability(0, 0) == pytest.approx(0.8)
    assert tm.sample_next_state(0, rng=np.random.default_rng(1)) in (0, 1)
    assert tm.n_step_matrix(2).shape == (2, 2)
    assert tm.expected_durations()[0] == pytest.approx(5.0)

    om = DiagonalGaussianObservationModel(means=[[-1.0], [1.0]], variances=[[0.25], [0.25]])
    assert om.n_states == 2
    assert om.emission_probability([-1.0], 0) > om.emission_probability([-1.0], 1)
    assert om.expected_observation(1)[0] == pytest.approx(1.0)
    y = om.sample_observation(0, rng=np.random.default_rng(0))
    assert y.shape == (1,)
    log_e = om.log_emission_matrix(np.linspace(-2, 2, 20))
    assert log_e.shape == (20, 2)
    resp = om.soft_responsibilities(np.linspace(-2, 2, 10), prior=[0.5, 0.5])
    assert resp.shape == (10, 2)
    assert om.predictive_density([-1.0], [0.9, 0.1]) > 0


@pytest.mark.unit
def test_forward_backward_occupancy_stable() -> None:
    rng = np.random.default_rng(2)
    t, k = 50, 3
    log_e = rng.normal(size=(t, k))
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    alpha, scales, ll = forward_probabilities(log_e, p)
    beta = backward_probabilities(log_e, p, scales=scales)
    gamma = state_occupancy_probabilities(alpha, beta)
    assert alpha.shape == (t, k)
    assert np.allclose(alpha.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(ll)
    xi = transition_probabilities(alpha, beta, log_e, p)
    assert xi.shape == (k, k)
    assert np.allclose(xi.sum(axis=1), 1.0, atol=1e-5)
    assert joint_probabilities(alpha, beta).shape == gamma.shape
    assert forecast_distribution(alpha[-1], p, 5).sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_mock_model_fit_filter_smooth_forecast(tmp_path: Path) -> None:
    frame = _synthetic_frame()
    model = MockDiscreteStateSpaceModel(n_states=3, random_seed=42)
    model.fit(frame, observation_columns=["close", "feature_a"])
    assert model.is_fitted

    filt = model.filter(frame, observation_columns=["close", "feature_a"])
    assert isinstance(filt, FilterResult)
    assert filt.n_steps == frame.height
    assert np.isfinite(filt.log_likelihood)

    smooth = model.smooth(frame, observation_columns=["close", "feature_a"])
    assert smooth.smoothed_probabilities.shape == filt.filtered_probabilities.shape

    lag_smooth = model.smooth(frame, observation_columns=["close"], lag=3)
    assert lag_smooth.metadata["algorithm"] == "fixed_lag"

    pred = model.predict(frame, observation_columns=["close"])
    proba = model.predict_proba(frame, observation_columns=["close"])
    assert pred.shape == (frame.height,)
    assert proba.shape[0] == frame.height

    fc = model.forecast(frame, horizon=4, observation_columns=["close"])
    assert isinstance(fc, ForecastResult)
    assert fc.horizon == 4
    assert len(fc.most_likely_path) == 4

    states, obs = model.sample(30, initial_state=0)
    assert states.shape == (30,)
    assert obs.shape[0] == 30

    ll = model.log_likelihood(frame, observation_columns=["close"])
    assert np.isfinite(ll)

    seq = model.state_sequence(frame, observation_columns=["close"])
    assert len(seq) == frame.height

    report = model.evaluate(frame, true_states=pred, observation_columns=["close"])
    assert "aic" in report["metrics"]
    assert report["metrics"]["state_prediction_accuracy"] == pytest.approx(1.0)

    path = model.save(tmp_path / "model.json")
    loaded = MockDiscreteStateSpaceModel.load(path)
    assert loaded.is_fitted
    assert loaded.log_likelihood(frame, observation_columns=["close"]) == pytest.approx(
        ll, rel=1e-5
    )


@pytest.mark.unit
def test_filters_smoothers_forecaster() -> None:
    rng = np.random.default_rng(3)
    log_e = rng.normal(size=(40, 2))
    p = np.array([[0.9, 0.1], [0.2, 0.8]])
    settings = StateSpaceSettings.default()
    settings = StateSpaceSettings.from_mapping(
        {
            **settings.model_dump(),
            "filtering": {"chunk_size": 10, "numerical_eps": 1e-300, "algorithm": "forward"},
        }
    )
    filt = ForwardFilter(settings).run(log_e, p)
    assert filt.metadata.get("chunked") is True
    smooth = FixedIntervalSmoother(settings).run(log_e, p, filter_result=filt)
    assert smooth.n_states == 2
    lag = FixedLagSmoother(settings).run(log_e, p, lag=4)
    assert lag.metadata["lag"] == 4
    fc = MultiStepForecaster(settings).forecast(filt.filtered_probabilities[-1], p, horizon=3)
    assert fc.step_distributions is not None


@pytest.mark.unit
def test_evaluation_and_diagnostics() -> None:
    pred = np.array([0, 0, 1, 1, 1, 0])
    proba = np.eye(2)[pred]
    tm = np.array([[0.7, 0.3], [0.4, 0.6]])
    metrics = EvaluationMetrics().evaluate(
        predicted=pred,
        probabilities=proba,
        log_likelihood=-10.0,
        n_params=4,
        n_samples=6,
        true_states=pred,
        transition_matrix=tm,
    )
    assert metrics["metrics"]["bic"] > 0
    y = np.linspace(-1, 1, 6)
    diag = StateSpaceDiagnostics().analyze(
        states=pred,
        probabilities=proba,
        transition_matrix=tm,
        observations=y,
        expected_observations=y,
        log_likelihood_history=[-12.0, -11.0, -10.5, -10.5],
        n_states=2,
    )
    assert diag["occupancy"]["counts"]
    assert diag["calibration"]["ece"] >= 0
    assert diag["residuals"]["available"] is True
    assert diag["likelihood_convergence"]["converged"] is True


@pytest.mark.unit
def test_storage_and_result_roundtrips(tmp_path: Path) -> None:
    filt = FilterResult(
        filtered_states=np.array([0, 1, 1]),
        filtered_probabilities=np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]]),
        log_likelihood=-1.5,
        normalization_constants=np.array([1.0, 0.9, 0.8]),
    )
    assert FilterResult.from_dict(filt.to_dict()).log_likelihood == -1.5
    smooth = FixedIntervalSmoother().run(
        np.log(np.clip(filt.filtered_probabilities, 1e-12, None)),
        [[0.8, 0.2], [0.3, 0.7]],
        filter_result=filt,
    )
    assert smoother_result_roundtrip(smooth)
    fc = ForecastResult.from_probabilities(
        np.array([0.2, 0.8]), horizon=2, expected_duration={0: 2.0, 1: 3.0}
    )
    assert ForecastResult.from_dict(fc.to_dict()).expected_state == 1

    store = StateStore(
        root=tmp_path / "ss",
        duckdb_path=tmp_path / "ss.duckdb",
        register_duckdb=True,
    )
    paths = store.write_filter_result(
        filt,
        model_name="mock_discrete_ssm",
        version="1.0.0",
        timestamps=[0, 1, 2],
        forecast=fc,
        diagnostics={"ok": True},
    )
    store.write_smoother_result(
        smooth, model_name="mock_discrete_ssm", version="1.0.0", timestamps=[0, 1, 2]
    )
    store.write_transition_matrix(
        np.array([[0.8, 0.2], [0.3, 0.7]]),
        model_name="mock_discrete_ssm",
        version="1.0.0",
    )
    assert paths["states"].exists()
    frame = store.read_states(
        exchange="synthetic",
        symbol="STATE",
        timeframe="1h",
        model_name="mock_discrete_ssm",
        version="1.0.0",
    )
    assert frame.height == 3
    assert store.stats()["file_count"] >= 1


def smoother_result_roundtrip(smooth: object) -> bool:
    from iqrp.app.state_space.base.smoother_result import SmootherResult

    assert isinstance(smooth, SmootherResult)
    return SmootherResult.from_dict(smooth.to_dict()).n_steps == smooth.n_steps
