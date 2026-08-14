"""Integration pipeline for Kalman filtering engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import iqrp.app.regimes.kalman
from iqrp.app.regimes.kalman import KalmanFilterModel, KalmanSettings, build_system, simulate_lds
from iqrp.app.regimes.kalman.prediction import prediction_intervals
from iqrp.app.regimes.kalman.visualization import (
    plot_covariance_evolution,
    plot_filtered_state,
    plot_innovations,
    plot_kalman_gain,
    plot_prediction_bands,
    plot_smoothed_state,
)
from iqrp.app.state_space import get_registry as get_ss_registry


@pytest.mark.integration
def test_kalman_end_to_end(tmp_path: Path) -> None:
    assert "kalman" in get_ss_registry().list_names()
    settings = KalmanSettings.from_hydra(
        overrides=[
            "application=trend",
            "filter_type=linear",
            f"output_dir={tmp_path / 'charts'}",
            "training.em_iterations=3",
            "training.estimate_noise=true",
        ]
    )
    sys = build_system(settings, application="trend")
    sys = type(sys)(
        f=sys.f,
        h=sys.h,
        q=sys.q,
        r=np.array([[1e-3]]),
        x0=sys.x0,
        p0=sys.p0,
        application="trend",
    )
    states, obs = simulate_lds(sys, 250, rng=np.random.default_rng(41))
    frame = pl.DataFrame({"open_time": list(range(obs.shape[0])), "close": obs[:, 0]})
    model = KalmanFilterModel(settings=settings, system=sys, random_seed=41)
    model.fit(frame, observation_columns=["close"])
    pred = model.predict(frame, observation_columns=["close"])
    proba = model.predict_proba(frame, observation_columns=["close"])
    fc = model.forecast(frame, observation_columns=["close"])
    report = model.evaluate(frame, true_states=states, observation_columns=["close"])
    diag = model.diagnostics(frame, observation_columns=["close"])
    charts = Path(settings.output_dir)
    charts.mkdir(parents=True, exist_ok=True)
    means = model.filtered_means()
    plot_filtered_state(means, charts / "filtered.svg", settings, observations=obs[:, 0])
    plot_smoothed_state(model.smoothed_means(), charts / "smoothed.svg", settings)
    lo, hi = prediction_intervals(means[-1], model._trace.covs[-1])  # type: ignore[union-attr]
    plot_prediction_bands(
        means[:, 0], means[:, 0] + lo[0], means[:, 0] + hi[0], charts / "bands.svg", settings
    )
    plot_innovations(model._trace.innovations, charts / "innov.svg", settings)  # type: ignore[union-attr]
    plot_covariance_evolution(model._trace.covs, charts / "cov.svg", settings)  # type: ignore[union-attr]
    plot_kalman_gain(model._trace.gains, charts / "gain.svg", settings)  # type: ignore[union-attr]
    path = model.save(tmp_path / "model.json")
    loaded = KalmanFilterModel.load(path)
    assert loaded.is_fitted
    assert report["metrics"]["state_corr"] >= 0.80
    assert fc.horizon == settings.forecasting.default_horizon
    assert pred.shape[0] == obs.shape[0]
    assert proba.shape[1] == 2
    assert "filter_stability" in diag
    assert (charts / "filtered.svg").exists()
