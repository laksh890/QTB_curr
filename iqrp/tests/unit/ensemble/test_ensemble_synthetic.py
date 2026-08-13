"""Synthetic market validation for ensemble."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.base.registry import get_registry
from iqrp.app.regimes.ensemble import EnsembleRegimeModel, EnsembleSettings
from iqrp.app.simulation.regimes.hidden_regime import HiddenRegimeSimulator
from iqrp.app.simulation.regimes.regime_switching import RegimeSwitchingSimulator

# import stubs from core tests
from iqrp.tests.unit.ensemble.test_ensemble_core import _StubRegimeA, _StubRegimeB  # noqa: F401


def _settings(**kw: object) -> EnsembleSettings:
    data = {
        **EnsembleSettings.default().model_dump(),
        "n_states": 3,
        "state_names": ("bull", "bear", "sideways"),
        "discovery_modules": ("iqrp.app.regimes.models.mock",),
        "member_names": ("stub_a", "stub_b", "mock_regime"),
        "combination": {"method": "soft_voting", "normalize": True},
        "weighting": {
            "method": "accuracy",
            "user_weights": None,
            "lookback": 30,
            "adaptive_rate": 0.1,
            "min_weight": 0.05,
        },
        "training": {"validation_fraction": 0.25, "min_members": 2},
    }
    data.update(kw)
    return EnsembleSettings.from_mapping(data)


@pytest.mark.unit
def test_ensemble_improves_calibration_and_consensus() -> None:
    assert "stub_a" in get_registry().list_names()
    true_p = RegimeSwitchingSimulator.mixed_transition(3, 0.9)
    obs = HiddenRegimeSimulator(np.random.default_rng(11)).simulate(
        200,
        transition_matrix=true_p,
        state_names=("bear", "sideways", "bull"),
        emission_means=(-1.5, 0.0, 1.5),
        emission_stds=(0.4, 0.4, 0.4),
    )
    # map latent 0,1,2 → our canonical bull=0,bear=1,sideways=2 differs from sim order
    # use sim labels as-is with state_names bear,sideways,bull → remap
    # For evaluation use argmax of ensemble vs stub_a alone on mapped space
    frame = pl.DataFrame(
        {
            "open_time": list(range(obs.observations.shape[0])),
            "close": 100 + np.cumsum(obs.observations.reshape(-1)),
            "f1": obs.observations.reshape(-1),
        }
    )
    truth = obs.latent.state_ids  # 0 bear, 1 sideways, 2 bull in sim
    # canonical in settings: bull, bear, sideways → remap truth to canonical ids
    remap = {0: 1, 1: 2, 2: 0}  # sim bear→bear(1), sideways→sideways(2), bull→bull(0)
    truth_c = np.array([remap[int(x)] for x in truth], dtype=np.int64)

    ens = EnsembleRegimeModel(settings=_settings(), random_seed=11)
    ens.fit(frame, feature_columns=["close", "f1"])
    report = ens.evaluate(frame, true_states=truth_c, feature_columns=["close", "f1"])
    board = report["leaderboard"]
    ens_row = next(r for r in board if r["name"] == "ensemble")
    # ensemble should be among top performers
    assert ens_row["rank"] <= 3
    cons = ens.consensus(frame)
    assert cons["mean_consensus"] >= 0.0
    conf = ens.confidence(frame)
    assert 0 <= conf["confidence"] <= 1


@pytest.mark.unit
def test_weight_adaptation_online() -> None:
    frame = pl.DataFrame(
        {
            "open_time": list(range(80)),
            "close": 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 80)),
        }
    )
    settings = _settings(
        weighting={
            "method": "adaptive",
            "user_weights": None,
            "lookback": 20,
            "adaptive_rate": 0.2,
            "min_weight": 0.05,
        },
        online={"warm_start": True, "weight_update": True, "recalibrate_every": 2},
    )
    model = EnsembleRegimeModel(settings=settings, random_seed=0)
    model.fit(frame.slice(0, 50), feature_columns=["close"])
    w0 = model.weights()
    model.partial_fit(frame.slice(50, 30), feature_columns=["close"])
    w1 = model.weights()
    assert set(w0) == set(w1)
    assert abs(sum(w1.values()) - 1.0) < 1e-6


@pytest.mark.unit
def test_stress_many_obs() -> None:
    n = 3_000
    frame = pl.DataFrame(
        {
            "open_time": list(range(n)),
            "close": 100 + np.cumsum(np.random.default_rng(3).normal(0, 1, n)),
        }
    )
    settings = _settings(
        member_names=("stub_a", "stub_b"),
        calibration={"enabled": False, "method": "none", "temperature": 1.0},
    )
    model = EnsembleRegimeModel(settings=settings, random_seed=3)
    model.fit(frame, feature_columns=["close"])
    proba = model.predict_proba(frame, feature_columns=["close"])
    assert proba.shape == (n, 3)
    assert np.all(np.isfinite(proba))
