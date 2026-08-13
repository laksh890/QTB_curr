"""Broad coverage for ensemble modules."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from iqrp.app.core.exceptions import ConfigurationError, ValidationError
from iqrp.app.regimes.ensemble.calibration import Calibrator, brier_score, _isotonic_regression
from iqrp.app.regimes.ensemble.combiner import stacking_combine
from iqrp.app.regimes.ensemble.config import EnsembleSettings
from iqrp.app.regimes.ensemble.confidence import credible_mass_interval, expected_persistence, forecast_uncertainty
from iqrp.app.regimes.ensemble.model import EnsembleRegimeModel, EnsembleStateSpaceModel, _as_frame
from iqrp.app.regimes.ensemble.registry import (
    EnsembleRegistry,
    build_state_map,
    discover_modules,
    list_available_members,
)
from iqrp.app.regimes.ensemble.serializer import _json_default
from iqrp.app.regimes.ensemble.weighting import (
    adaptive_update,
    calibration_weights,
    log_likelihood_weights,
    recent_accuracy_weights,
    rolling_weights,
    stability_weights,
)
from iqrp.app.regimes.ensemble.visualization import _ensure, plot_member_timelines, plot_transition_chart
from iqrp.tests.unit.ensemble.test_ensemble_core import _StubRegimeA, _StubRegimeB  # noqa: F401


def _settings(**kw: object) -> EnsembleSettings:
    data = {
        **EnsembleSettings.default().model_dump(),
        "n_states": 3,
        "state_names": ("bull", "bear", "sideways"),
        "member_names": ("stub_a", "stub_b"),
        "discovery_modules": ("iqrp.app.regimes.models.mock",),
        "training": {"validation_fraction": 0.2, "min_members": 1},
    }
    data.update(kw)
    return EnsembleSettings.from_mapping(data)


@pytest.mark.unit
def test_config_and_discovery() -> None:
    s = EnsembleSettings.default()
    assert s.enabled
    s2 = EnsembleSettings.from_mapping(OmegaConf.create({"n_states": 4}))
    assert s2.n_states == 4
    with pytest.raises(ConfigurationError):
        EnsembleSettings.from_mapping("x")  # type: ignore[arg-type]
    bad = Path("/tmp/ens_bad.yaml")
    bad.write_text("- a\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        EnsembleSettings.from_hydra(bad)
    with patch(
        "iqrp.app.regimes.ensemble.config._default_config_path",
        return_value=Path("/tmp/missing_ens.yaml"),
    ):
        assert EnsembleSettings.default().n_states == 6
    loaded = discover_modules(("iqrp.app.regimes.models.mock", "no.such.module.xyz"))
    assert "iqrp.app.regimes.models.mock" in loaded
    assert "ensemble" not in list_available_members()
    smap = build_state_map(("bearish", "bullish"), ("bull", "bear", "sideways"))
    assert smap.shape == (2, 3)


@pytest.mark.unit
def test_weighting_and_confidence_helpers() -> None:
    names = ["a", "b"]
    preds = {"a": np.array([0, 1, 1]), "b": np.array([0, 0, 1])}
    truth = np.array([0, 1, 1])
    assert recent_accuracy_weights(preds, truth, names, lookback=2).sum() == pytest.approx(1)
    assert log_likelihood_weights({"a": -1, "b": -2}, names).sum() == pytest.approx(1)
    assert calibration_weights({"a": 0.1, "b": 0.5}, names).sum() == pytest.approx(1)
    ph = {"a": np.eye(3), "b": np.eye(3)}
    assert stability_weights(ph, names).sum() == pytest.approx(1)
    assert rolling_weights(np.array([[1, 0], [0.5, 0.5]])).sum() == pytest.approx(1)
    assert adaptive_update(np.array([0.5, 0.5]), np.array([1.0, 0.0])).sum() == pytest.approx(1)
    lo, hi = credible_mass_interval(np.array([0.1, 0.7, 0.2]))
    assert lo <= hi
    assert expected_persistence(np.eye(3) * 0.8 + 0.1, 0) > 1
    assert forecast_uncertainty(np.array([[0.5, 0.5], [0.9, 0.1]])).shape[0] == 2
    mw = np.array([[0.6, 0.4], [0.3, 0.7]])
    out = stacking_combine(
        [np.array([[0.5, 0.5], [0.2, 0.8]]), np.array([[0.4, 0.6], [0.3, 0.7]])],
        mw,
    )
    assert out.shape == (2, 2)


@pytest.mark.unit
def test_calibrator_roundtrip_and_brier() -> None:
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    y = np.array([0, 2, 1])
    cal = Calibrator(method="isotonic").fit(p, y)
    d = cal.to_dict()
    cal2 = Calibrator.from_dict(d)
    assert cal2.transform(p).shape == p.shape
    assert brier_score(p, y) >= 0
    assert _isotonic_regression(np.array([])).size == 0
    assert _isotonic_regression(np.array([0.1, 0.4, 0.2, 0.8])).size == 4


@pytest.mark.unit
def test_model_errors_and_ssm_sample(tmp_path: Path) -> None:
    model = EnsembleRegimeModel(settings=_settings())
    with pytest.raises(ValidationError):
        model.predict_proba(pl.DataFrame({"close": [1.0]}))
    # no members
    with pytest.raises(ConfigurationError):
        EnsembleRegistry(
            _settings(member_names=("nope",), training={"validation_fraction": 0.2, "min_members": 1})
        ).create_members()
    frame = pl.DataFrame({"close": np.linspace(1, 2, 40)})
    model.fit(frame, feature_columns=["close"])
    # warm_start false
    s2 = _settings(online={"warm_start": False, "weight_update": False, "recalibrate_every": 0})
    m2 = EnsembleRegimeModel(settings=s2)
    m2.fit(frame, feature_columns=["close"])
    m2.partial_fit(frame.slice(30, 10), feature_columns=["close"])
    assert isinstance(_json_default(np.array([1.0])), list)
    assert isinstance(_json_default(np.bool_(True)), bool)
    with pytest.raises(TypeError):
        _json_default(object())
    off = _settings(visualization={"enabled": False, "max_points": 10})
    _ensure(tmp_path / "x.svg", off)
    plot_member_timelines({"a": np.arange(5)}, tmp_path / "mt.svg", off)
    plot_transition_chart(np.eye(2), tmp_path / "tr.svg")
    ssm = EnsembleStateSpaceModel(settings=_settings())
    ssm.fit(frame, observation_columns=["close"])
    st, ob = ssm.sample(10, initial_state=0)
    assert st.shape[0] == 10 and ob.shape[0] == 10
    assert np.isfinite(ssm.log_likelihood(frame, observation_columns=["close"]))
    # ndarray fit path
    arr = frame["close"].to_numpy()
    ssm2 = EnsembleStateSpaceModel(settings=_settings())
    ssm2.fit(arr, observation_columns=["close"])
    assert _as_frame(arr, ["close"]).height == arr.size
