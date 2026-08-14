"""Integration pipeline for HMM engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import iqrp.app.regimes.hmm
from iqrp.app.regimes.hmm import HiddenMarkovModel, HMMSettings
from iqrp.app.regimes.hmm.visualization import (
    plot_hidden_state_timeline,
    plot_likelihood_curve,
    plot_posterior_heatmap,
    plot_transition_heatmap,
)
from iqrp.app.simulation.regimes.hidden_regime import HiddenRegimeSimulator
from iqrp.app.simulation.regimes.regime_switching import RegimeSwitchingSimulator
from iqrp.app.state_space import get_registry as get_ss_registry


@pytest.mark.integration
def test_hmm_end_to_end(tmp_path: Path) -> None:
    assert "hmm" in get_ss_registry().list_names()
    true_p = RegimeSwitchingSimulator.mixed_transition(2, 0.9)
    obs = HiddenRegimeSimulator(np.random.default_rng(21)).simulate(
        600,
        transition_matrix=true_p,
        state_names=("bearish", "bullish"),
        emission_means=(-1.2, 1.2),
        emission_stds=(0.3, 0.3),
    )
    frame = pl.DataFrame(
        {"open_time": list(range(len(obs.observations))), "ret": obs.observations.tolist()}
    )
    settings = HMMSettings.from_hydra(
        overrides=[
            "n_states=2",
            f"output_dir={tmp_path / 'charts'}",
            "training.max_iter=60",
            "initialization.n_restarts=2",
        ]
    )
    model = HiddenMarkovModel(n_states=2, settings=settings, random_seed=21)
    model.fit(frame, observation_columns=["ret"])
    pred = model.predict(frame, observation_columns=["ret"])
    proba = model.predict_proba(frame, observation_columns=["ret"])
    fc = model.forecast(frame, observation_columns=["ret"])
    report = model.evaluate(frame, true_states=obs.latent.state_ids, observation_columns=["ret"])
    diag = model.diagnostics(frame, observation_columns=["ret"])
    charts = Path(settings.output_dir)
    charts.mkdir(parents=True, exist_ok=True)
    plot_hidden_state_timeline(pred, charts / "states.svg", settings)
    plot_posterior_heatmap(proba, charts / "posterior.svg", settings)
    plot_transition_heatmap(model.transition_matrix(), charts / "trans.svg", settings)
    plot_likelihood_curve(diag["convergence"]["history"], charts / "ll.svg", settings)
    path = model.save(tmp_path / "model.json")
    loaded = HiddenMarkovModel.load(path)
    assert loaded.is_fitted
    assert report["metrics"]["prediction_accuracy"] >= 0.75
    assert fc.horizon == settings.forecasting.default_horizon
    assert (charts / "states.svg").exists()
