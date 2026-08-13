"""Integration pipeline: fit → filter → smooth → forecast → store → charts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.state_space import (
    StateSpaceDiagnostics,
    StateSpaceSettings,
    StateStore,
    get_registry,
)
from iqrp.app.state_space.visualization import (
    plot_forecast_uncertainty,
    plot_persistence_distribution,
    plot_probability_heatmap,
    plot_state_timeline,
    plot_transition_graph,
)


@pytest.mark.integration
def test_state_space_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n = 120
    latent = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
    y = np.where(latent == 0, rng.normal(-1.0, 0.25, n), rng.normal(1.0, 0.25, n))
    frame = pl.DataFrame({"open_time": list(range(n)), "close": y, "feat": y + 0.01})

    settings = StateSpaceSettings.from_hydra(
        overrides=[
            f"store_dir={tmp_path / 'store'}",
            f"duckdb_path={tmp_path / 'ss.duckdb'}",
            f"output_dir={tmp_path / 'charts'}",
            "forecasting.default_horizon=6",
        ]
    )
    model = get_registry().create("mock_discrete_ssm", n_states=2, settings=settings)
    model.fit(frame)

    filt = model.filter(frame)
    smooth = model.smooth(frame)
    fc = model.forecast(frame)
    report = model.evaluate(frame, true_states=latent)
    diag = StateSpaceDiagnostics().analyze(
        states=filt.filtered_states,
        probabilities=filt.filtered_probabilities,
        transition_matrix=model._transition_matrix_or_none(),
        log_likelihood_history=[filt.log_likelihood - 1.0, filt.log_likelihood],
    )

    store = StateStore(settings=settings)
    paths = store.write_filter_result(
        filt,
        model_name=model.meta.name,
        version=model.meta.version,
        timestamps=frame["open_time"].to_list(),
        forecast=fc,
        diagnostics=diag,
        metadata={"eval": report["metrics"]},
    )
    store.write_transition_matrix(
        model._transition_matrix_or_none(),  # type: ignore[arg-type]
        model_name=model.meta.name,
        version=model.meta.version,
    )
    store.write_smoother_result(
        smooth,
        model_name=model.meta.name,
        version=model.meta.version,
        timestamps=frame["open_time"].to_list(),
    )

    charts = Path(settings.output_dir)
    charts.mkdir(parents=True, exist_ok=True)
    plot_state_timeline(filt.filtered_states, charts / "timeline.svg", settings)
    plot_probability_heatmap(filt.filtered_probabilities, charts / "proba.svg", settings)
    plot_transition_graph(
        model._transition_matrix_or_none(),  # type: ignore[arg-type]
        charts / "transitions.svg",
        settings,
        state_names=model.meta.state_names,
    )
    plot_persistence_distribution(
        diag["persistence"]["run_lengths"], charts / "persist.svg", settings
    )
    assert fc.step_distributions is not None
    plot_forecast_uncertainty(fc.step_distributions, charts / "forecast.svg", settings)

    assert paths["states"].exists()
    assert (charts / "timeline.svg").exists()
    assert (
        store.read_states(
            exchange="synthetic",
            symbol="STATE",
            timeframe="1h",
            model_name=model.meta.name,
        ).height
        == n
    )
    # Synthetic recovery should beat chance for 2 well-separated Gaussians
    assert report["metrics"]["state_prediction_accuracy"] >= 0.7
