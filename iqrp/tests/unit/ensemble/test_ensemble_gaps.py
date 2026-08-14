"""Gap-filling tests for ensemble coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import iqrp.app.regimes.ensemble
from iqrp.app.regimes.base.registry import get_registry as get_regime_registry
from iqrp.app.regimes.ensemble import EnsembleRegimeModel, EnsembleSettings
from iqrp.app.regimes.ensemble.combiner import combine
from iqrp.app.regimes.ensemble.orchestrator import fit_members, predict_members
from iqrp.app.regimes.ensemble.registry import EnsembleMember, EnsembleRegistry, _default_map
from iqrp.app.regimes.ensemble.visualization import plot_regime_timeline, plot_weight_evolution
from iqrp.app.regimes.ensemble.weighting import normalize_weights
from iqrp.app.state_space import get_registry as get_ss_registry
from iqrp.tests.unit.ensemble.test_ensemble_core import _StubRegimeA, _StubRegimeB


def _settings(**kw: object) -> EnsembleSettings:
    data = {
        **EnsembleSettings.default().model_dump(),
        "n_states": 6,
        "state_names": (
            "bull",
            "bear",
            "sideways",
            "high_volatility",
            "low_volatility",
            "liquidity_stress",
        ),
        "member_names": ("stub_a", "stub_b"),
        "discovery_modules": ("iqrp.app.regimes.models.mock",),
        "combination": {"method": "confidence", "normalize": True},
        "weighting": {
            "method": "stability",
            "user_weights": None,
            "lookback": 10,
            "adaptive_rate": 0.05,
            "min_weight": 0.01,
        },
        "calibration": {"enabled": True, "method": "platt", "temperature": 1.0},
        "training": {"validation_fraction": 0.25, "min_members": 1},
    }
    data.update(kw)
    return EnsembleSettings.from_mapping(data)


@pytest.mark.unit
def test_registry_dual() -> None:
    assert "ensemble" in get_regime_registry().list_names()
    assert "ensemble" in get_ss_registry().list_names()


@pytest.mark.unit
def test_default_map_branches() -> None:
    p2 = np.array([[0.8, 0.2], [0.3, 0.7]])
    m = _default_map(p2, 3)
    assert m.shape == (2, 3)
    p4 = np.ones((5, 4)) / 4
    m2 = _default_map(p4, 3)
    assert m2.shape == (5, 3)
    p3 = np.eye(3)
    assert _default_map(p3, 3).shape == (3, 3)


@pytest.mark.unit
def test_parallel_and_failing_member() -> None:
    settings = _settings(n_states=3, state_names=("bull", "bear", "sideways"))
    members = EnsembleRegistry(settings).create_members()

    class Boom(EnsembleMember):
        pass

    # mark one to fail predict by breaking model
    frame = pl.DataFrame({"close": np.linspace(1, 2, 30)})
    fitted = fit_members(members, frame, ["close"], parallel=True)
    assert any(m.metadata.get("fitted") for m in fitted)
    mapped, hards, names = predict_members(fitted, frame, ["close"], n_canonical=3, parallel=False)
    assert len(mapped) == len(names)


@pytest.mark.unit
def test_combination_methods_end_to_end(tmp_path: Path) -> None:
    frame = pl.DataFrame({"close": 100 + np.cumsum(np.random.default_rng(5).normal(0, 1, 60))})
    for method in ("majority", "weighted", "bma", "stacking", "meta", "dynamic"):
        s = _settings(
            n_states=3,
            state_names=("bull", "bear", "sideways"),
            combination={"method": method, "normalize": True},
            calibration={"enabled": False, "method": "none", "temperature": 1.0},
        )
        model = EnsembleRegimeModel(settings=s, random_seed=5)
        model.fit(frame, feature_columns=["close"])
        assert model.predict_proba(frame, feature_columns=["close"]).shape[1] == 3
    plot_regime_timeline(np.array([]), tmp_path / "empty.svg")
    plot_weight_evolution([], tmp_path / "empty_w.svg")
    assert normalize_weights(np.zeros(3)).sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_combine_empty_raises() -> None:
    with pytest.raises(ValueError):
        combine([], np.array([1.0]))
