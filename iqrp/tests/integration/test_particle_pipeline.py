"""Integration pipeline for particle filter engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import iqrp.app.regimes.particle
from iqrp.app.regimes.particle import (
    ParticleFilterModel,
    ParticleSettings,
    build_transition,
    simulate_nonlinear,
)
from iqrp.app.regimes.particle.visualization import (
    plot_ess_timeline,
    plot_particle_cloud,
    plot_state_trajectory,
    plot_weight_histogram,
)
from iqrp.app.state_space import get_registry as get_ss_registry


@pytest.mark.integration
def test_particle_end_to_end(tmp_path: Path) -> None:
    assert "particle" in get_ss_registry().list_names()
    settings = ParticleSettings.from_hydra(
        overrides=[
            "application=nonlinear_trend",
            "filter_type=bootstrap",
            "n_particles=120",
            f"output_dir={tmp_path / 'charts'}",
            "system.observation_noise_scale=0.04",
            "system.process_noise_scale=0.05",
            "rejuvenation.enabled=true",
        ]
    )
    model = build_transition(settings, application="nonlinear_trend")
    states, obs = simulate_nonlinear(model, 100, rng=np.random.default_rng(41), obs_scale=0.04)
    frame = pl.DataFrame({"open_time": list(range(obs.shape[0])), "close": obs[:, 0]})
    pf = ParticleFilterModel(settings=settings, transition=model, random_seed=41)
    pf.fit(frame, observation_columns=["close"])
    pred = pf.predict(frame, observation_columns=["close"])
    proba = pf.predict_proba(frame, observation_columns=["close"])
    fc = pf.forecast(frame, observation_columns=["close"])
    report = pf.evaluate(frame, true_states=states, observation_columns=["close"])
    diag = pf.diagnostics(frame, observation_columns=["close"])
    charts = Path(settings.output_dir)
    charts.mkdir(parents=True, exist_ok=True)
    plot_state_trajectory(
        pf.filtered_means(), charts / "traj.svg", settings, observations=obs[:, 0]
    )
    plot_particle_cloud(pf._cloud.states, pf._cloud.weights, charts / "cloud.svg", settings)  # type: ignore[union-attr]
    plot_weight_histogram(pf._cloud.weights, charts / "weights.svg", settings)  # type: ignore[union-attr]
    plot_ess_timeline(pf._trace.ess, charts / "ess.svg", settings)  # type: ignore[union-attr]
    path = pf.save(tmp_path / "model.json")
    loaded = ParticleFilterModel.load(path)
    assert loaded.is_fitted
    assert report["metrics"]["state_corr"] >= 0.50
    assert fc.horizon == settings.forecasting.default_horizon
    assert pred.shape[0] == obs.shape[0]
    assert proba.shape[1] == 2
    assert "ess" in diag
    assert (charts / "traj.svg").exists()
