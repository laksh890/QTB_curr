"""Integration: Markov engine + simulation + state-space registry + charts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import iqrp.app.regimes.markov
from iqrp.app.regimes.markov import (
    MarkovChainModel,
    MarkovSettings,
    MarkovTrainer,
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
from iqrp.app.state_space import get_registry as get_ss_registry


@pytest.mark.integration
def test_markov_end_to_end_pipeline(tmp_path: Path) -> None:
    assert "markov_chain" in get_ss_registry().list_names()

    k = 3
    true_p = RegimeSwitchingSimulator.mixed_transition(k, 0.9)
    path = RegimeSwitchingSimulator(np.random.default_rng(11)).simulate(
        800,
        transition_matrix=true_p,
        state_names=("alpha", "beta", "gamma"),
        drifts=(0.05, 0.0, -0.05),
        volatilities=(0.2, 0.15, 0.25),
    )
    frame = pl.DataFrame(
        {
            "open_time": list(range(len(path.state_ids))),
            "state_id": path.state_ids.tolist(),
        }
    )
    settings = MarkovSettings.from_hydra(
        overrides=[
            "n_states=3",
            f"output_dir={tmp_path / 'charts'}",
            "forecasting.default_horizon=6",
        ]
    )
    model = MarkovChainModel(
        n_states=3,
        state_names=("alpha", "beta", "gamma"),
        settings=settings,
        random_seed=11,
    )
    stats = MarkovTrainer(settings).train(model, frame, true_states=path.state_ids)
    assert stats["evaluation"]["metrics"]["prediction_accuracy"] >= 0.99

    fc = model.forecast(frame)
    charts = Path(settings.output_dir)
    charts.mkdir(parents=True, exist_ok=True)
    plot_transition_heatmap(
        model.transition_matrix(), charts / "heatmap.svg", settings, state_names=model.state_names
    )
    plot_transition_graph(
        model.transition_matrix(), charts / "graph.svg", settings, state_names=model.state_names
    )
    plot_occupancy_timeline(path.state_ids, charts / "timeline.svg", settings)
    persist = model.persistence_report(frame)
    plot_persistence_histogram(persist["run_lengths"], charts / "persist.svg", settings)
    assert fc.step_distributions is not None
    plot_forecast_probabilities(fc.step_distributions, charts / "forecast.svg", settings)
    plot_stationary_distribution(
        model.stationary_distribution(),
        charts / "stationary.svg",
        settings,
        state_names=model.state_names,
    )
    path_model = model.save(tmp_path / "model.json")
    loaded = MarkovChainModel.load(path_model)
    assert np.allclose(loaded.transition_matrix(), model.transition_matrix())
    assert (charts / "heatmap.svg").exists()
    assert np.linalg.norm(model.transition_matrix() - true_p) < 0.25
