"""Integration pipeline for Bayesian Regime Switching Engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import iqrp.app.regimes.bayesian  # noqa: F401
from iqrp.app.regimes.bayesian import BayesianRegimeSwitchingModel, BayesianSettings
from iqrp.app.regimes.bayesian.visualization import (
    plot_regime_timeline,
    plot_trace,
    plot_transition_uncertainty,
)
from iqrp.app.simulation.regimes.hidden_regime import HiddenRegimeSimulator
from iqrp.app.simulation.regimes.regime_switching import RegimeSwitchingSimulator
from iqrp.app.state_space import get_registry as get_ss_registry


@pytest.mark.integration
def test_bayesian_end_to_end(tmp_path: Path) -> None:
    assert "bayesian_regime" in get_ss_registry().list_names()
    true_p = RegimeSwitchingSimulator.mixed_transition(2, 0.9)
    obs = HiddenRegimeSimulator(np.random.default_rng(31)).simulate(
        400,
        transition_matrix=true_p,
        state_names=("bearish", "bullish"),
        emission_means=(-1.2, 1.2),
        emission_stds=(0.3, 0.3),
    )
    frame = pl.DataFrame(
        {"open_time": list(range(len(obs.observations))), "ret": obs.observations.tolist()}
    )
    settings = BayesianSettings.from_hydra(
        overrides=[
            "n_states=2",
            f"output_dir={tmp_path / 'charts'}",
            "inference.algorithm=gibbs",
            "inference.n_samples=60",
            "inference.burn_in=15",
            "inference.n_chains=2",
            "inference.n_jobs=2",
        ]
    )
    model = BayesianRegimeSwitchingModel(n_states=2, settings=settings, random_seed=31)
    model.fit(frame, observation_columns=["ret"])
    pred = model.predict(frame, observation_columns=["ret"])
    proba = model.predict_proba(frame, observation_columns=["ret"])
    fc = model.forecast(frame, observation_columns=["ret"])
    report = model.evaluate(frame, true_states=obs.latent.state_ids, observation_columns=["ret"])
    diag = model.diagnostics(frame, observation_columns=["ret"])
    charts = Path(settings.output_dir)
    charts.mkdir(parents=True, exist_ok=True)
    plot_regime_timeline(proba, charts / "states.svg", settings)
    plot_trace(diag["history"], charts / "trace.svg", settings)
    ci = model.credible_intervals("transition")
    plot_transition_uncertainty(
        model.transition_matrix(), ci["low"], ci["high"], charts / "trans.svg", settings
    )
    path = model.save(tmp_path / "model.json")
    loaded = BayesianRegimeSwitchingModel.load(path)
    assert loaded.is_fitted
    assert report["metrics"]["prediction_accuracy"] >= 0.70
    assert fc.horizon == settings.forecasting.default_horizon
    assert pred.shape[0] == frame.height
    assert (charts / "states.svg").exists()
