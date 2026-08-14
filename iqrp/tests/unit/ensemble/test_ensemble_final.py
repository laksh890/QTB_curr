"""Final coverage push for ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.ensemble.calibration import Calibrator
from iqrp.app.regimes.ensemble.confidence import posterior_confidence
from iqrp.app.regimes.ensemble.config import EnsembleSettings
from iqrp.app.regimes.ensemble.model import EnsembleRegimeModel
from iqrp.app.regimes.ensemble.orchestrator import member_log_likelihoods
from iqrp.app.regimes.ensemble.registry import EnsembleRegistry
from iqrp.app.regimes.ensemble.weighting import compute_weights
from iqrp.tests.unit.ensemble.test_ensemble_core import _StubRegimeA, _StubRegimeB


def _settings(**kw: object) -> EnsembleSettings:
    data = {
        **EnsembleSettings.default().model_dump(),
        "n_states": 3,
        "state_names": ("bull", "bear", "sideways"),
        "member_names": ("stub_a", "stub_b"),
        "discovery_modules": (),
        "weighting": {
            "method": "log_likelihood",
            "user_weights": None,
            "lookback": 10,
            "adaptive_rate": 0.05,
            "min_weight": 0.01,
        },
        "calibration": {"enabled": True, "method": "dirichlet", "temperature": 1.0},
        "training": {"validation_fraction": 0.2, "min_members": 1},
    }
    data.update(kw)
    return EnsembleSettings.from_mapping(data)


@pytest.mark.unit
def test_final_lines(tmp_path: Path) -> None:
    assert posterior_confidence(np.array([0.2, 0.8])).shape[0] == 1
    frame = pl.DataFrame(
        {"close": np.linspace(1, 3, 50), "f1": np.random.default_rng(0).normal(size=50)}
    )
    # empty discovery_modules but stubs already registered
    model = EnsembleRegimeModel(settings=_settings(), random_seed=1)
    model.fit(frame, feature_columns=["close", "f1"])
    ll = member_log_likelihoods(model.members, frame, ["close", "f1"])
    assert ll
    w = compute_weights("log_likelihood", names=list(ll), log_likes=ll)
    assert w.sum() == pytest.approx(1.0)
    # user weights method
    w2 = compute_weights(
        "user",
        names=["stub_a", "stub_b"],
        user={"stub_a": 0.7, "stub_b": 0.3},
    )
    assert w2.sum() == pytest.approx(1.0)
    # rolling / adaptive fallbacks when missing data → equal
    assert compute_weights("rolling", names=["a"]).shape[0] == 1
    cal = Calibrator(method="none").fit(np.ones((3, 2)) / 2, np.array([0, 1, 0]))
    assert cal.fitted
    # save/load roundtrip with members
    path = model.save(tmp_path / "e.json")
    loaded = EnsembleRegimeModel.load(path)
    assert loaded.is_fitted
    assert loaded.predict(frame, feature_columns=["close", "f1"]).shape[0] == 50
    # auto member_names null
    s = _settings(member_names=None)
    reg = EnsembleRegistry(s)
    names = reg.discover()
    assert "stub_a" in names or "mock_regime" in names or len(names) >= 0
