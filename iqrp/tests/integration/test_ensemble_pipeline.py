"""Integration pipeline for ensemble regime engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import iqrp.app.regimes.ensemble
from iqrp.app.regimes.ensemble import EnsembleRegimeModel, EnsembleSettings
from iqrp.app.regimes.ensemble.visualization import (
    plot_confidence_timeline,
    plot_probability_dashboard,
    plot_regime_timeline,
)
from iqrp.app.state_space import get_registry as get_ss_registry
from iqrp.tests.unit.ensemble.test_ensemble_core import _StubRegimeA, _StubRegimeB


@pytest.mark.integration
def test_ensemble_end_to_end(tmp_path: Path) -> None:
    assert "ensemble" in get_ss_registry().list_names()
    settings = EnsembleSettings.from_hydra(
        overrides=[
            "n_states=3",
            "state_names=[bull,bear,sideways]",
            "member_names=[stub_a,stub_b,mock_regime]",
            "combination.method=soft_voting",
            "weighting.method=accuracy",
            "calibration.method=temperature",
            f"output_dir={tmp_path / 'charts'}",
        ]
    )
    rng = np.random.default_rng(41)
    n = 120
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    frame = pl.DataFrame(
        {"open_time": list(range(n)), "close": close, "f1": np.diff(close, prepend=0)}
    )
    model = EnsembleRegimeModel(settings=settings, random_seed=41)
    model.fit(frame, feature_columns=["close", "f1"])
    pred = model.predict(frame, feature_columns=["close", "f1"])
    proba = model.predict_proba(frame, feature_columns=["close", "f1"])
    fc = model.forecast(frame, steps=settings.forecasting.default_horizon)
    board = model.leaderboard(frame, true_states=pred)
    diag = model.diagnostics(frame, true_states=pred)
    charts = Path(settings.output_dir)
    charts.mkdir(parents=True, exist_ok=True)
    plot_regime_timeline(pred, charts / "timeline.svg", settings)
    plot_confidence_timeline(proba.max(axis=1), charts / "conf.svg", settings)
    plot_probability_dashboard(proba, model._state_names, charts / "proba.svg", settings)
    path = model.save(tmp_path / "model.json")
    loaded = EnsembleRegimeModel.load(path)
    assert loaded.is_fitted
    assert proba.shape == (n, 3)
    assert fc.steps == settings.forecasting.default_horizon
    assert board[0]["name"]
    assert "weights" in diag
    assert (charts / "timeline.svg").exists()
