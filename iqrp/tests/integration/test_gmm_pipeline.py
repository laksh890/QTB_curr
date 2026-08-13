"""Integration pipeline for GMM regime engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import iqrp.app.regimes.gmm  # noqa: F401
from iqrp.app.regimes.gmm import GaussianMixtureModel, GMMSettings
from iqrp.app.regimes.gmm.visualization import (
    plot_cluster_scatter,
    plot_likelihood_curve,
    plot_probability_heatmap,
)
from iqrp.app.state_space import get_registry as get_ss_registry


@pytest.mark.integration
def test_gmm_end_to_end(tmp_path: Path) -> None:
    assert "gmm" in get_ss_registry().list_names()
    rng = np.random.default_rng(41)
    n = 300
    labels = rng.choice(2, size=n, p=[0.5, 0.5])
    x = np.empty((n, 2))
    x[labels == 0] = rng.normal([-1.2, -1.2], 0.3, size=(np.sum(labels == 0), 2))
    x[labels == 1] = rng.normal([1.2, 1.2], 0.3, size=(np.sum(labels == 1), 2))
    frame = pl.DataFrame({"open_time": list(range(n)), "f1": x[:, 0], "f2": x[:, 1]})
    settings = GMMSettings.from_hydra(
        overrides=[
            "n_components=2",
            f"output_dir={tmp_path / 'charts'}",
            "training.max_iter=60",
            "initialization.n_restarts=2",
            "covariance.type=full",
        ]
    )
    model = GaussianMixtureModel(n_components=2, settings=settings, random_seed=41)
    model.fit(frame, observation_columns=["f1", "f2"])
    pred = model.predict(frame, observation_columns=["f1", "f2"])
    proba = model.predict_proba(frame, observation_columns=["f1", "f2"])
    fc = model.forecast(frame, observation_columns=["f1", "f2"])
    report = model.evaluate(frame, true_states=labels, observation_columns=["f1", "f2"])
    diag = model.diagnostics(frame, observation_columns=["f1", "f2"])
    charts = Path(settings.output_dir)
    charts.mkdir(parents=True, exist_ok=True)
    plot_cluster_scatter(x, pred, charts / "scatter.svg", settings)
    plot_probability_heatmap(proba, charts / "posterior.svg", settings)
    plot_likelihood_curve(diag["history"], charts / "ll.svg", settings)
    path = model.save(tmp_path / "model.json")
    loaded = GaussianMixtureModel.load(path)
    assert loaded.is_fitted
    assert report["metrics"]["prediction_accuracy"] >= 0.80
    assert fc.horizon == settings.forecasting.default_horizon
    assert (charts / "scatter.svg").exists()
