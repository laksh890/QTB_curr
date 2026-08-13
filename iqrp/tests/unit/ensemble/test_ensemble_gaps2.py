"""Additional coverage gaps for ensemble (>98%)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from iqrp.app.core.exceptions import ValidationError
from iqrp.app.regimes.base.regime_model import RegimeModel, RegimeModelMeta
from iqrp.app.regimes.ensemble.calibration import Calibrator, expected_calibration_error
from iqrp.app.regimes.ensemble.confidence import expected_persistence, forecast_uncertainty, posterior_confidence
from iqrp.app.regimes.ensemble.config import EnsembleSettings
from iqrp.app.regimes.ensemble.disagreement import pairwise_disagreement
from iqrp.app.regimes.ensemble.evaluator import EnsembleEvaluator
from iqrp.app.regimes.ensemble.model import EnsembleRegimeModel, EnsembleStateSpaceModel
from iqrp.app.regimes.ensemble.orchestrator import (
    collect_transition,
    fit_members,
    member_log_likelihoods,
    predict_members,
)
from iqrp.app.regimes.ensemble.registry import EnsembleMember, EnsembleRegistry, build_state_map
from iqrp.app.regimes.ensemble.serializer import EnsembleSerializer
from iqrp.app.regimes.ensemble.trainer import EnsembleTrainer
from iqrp.app.regimes.ensemble.weighting import compute_weights, stability_weights, user_weights
from iqrp.app.regimes.ensemble.visualization import plot_agreement_heatmap, plot_probability_dashboard
from iqrp.tests.unit.ensemble.test_ensemble_core import _StubRegimeA, _StubRegimeB  # noqa: F401


def _settings(**kw: object) -> EnsembleSettings:
    data = {
        **EnsembleSettings.default().model_dump(),
        "n_states": 3,
        "state_names": ("bull", "bear", "sideways"),
        "member_names": ("stub_a", "stub_b"),
        "discovery_modules": (),
        "training": {"validation_fraction": 0.2, "min_members": 1},
        "online": {"warm_start": True, "weight_update": True, "recalibrate_every": 1},
        "calibration": {"enabled": True, "method": "isotonic", "temperature": 1.0},
        "columns": {"timestamp": "open_time", "feature_columns": ("close",)},
    }
    data.update(kw)
    return EnsembleSettings.from_mapping(data)


@pytest.mark.unit
def test_orchestrator_failure_paths() -> None:
    settings = _settings()
    members = EnsembleRegistry(settings).create_members()
    # inject a broken member
    class BadModel:
        is_fitted = False
        meta = RegimeModelMeta(
            name="bad", version="1", description="x", n_states=2, algorithm_family="x"
        )

        def fit(self, *a, **k):
            raise RuntimeError("boom")

        def predict_proba(self, *a, **k):
            raise RuntimeError("boom")

        def transition_matrix(self):
            raise RuntimeError("boom")

    bad = EnsembleMember(name="bad", model=BadModel())  # type: ignore[arg-type]
    frame = pl.DataFrame({"close": np.linspace(1, 2, 20)})
    out = fit_members(members + [bad], frame, ["close"], parallel=False)
    assert any(m.metadata.get("error") for m in out)
    # parallel fit
    out2 = fit_members(members, frame, ["close"], parallel=True)
    assert out2
    mapped, _, names = predict_members(out2, frame, ["close"], n_canonical=3, parallel=True)
    assert len(mapped) == len(names)
    # all fail predict
    for m in out2:
        m.metadata["fitted"] = False
    with pytest.raises(ValidationError):
        predict_members(out2, frame, ["close"], n_canonical=3, parallel=False)
    # LL failures
    for m in members:
        m.metadata["fitted"] = True
    with patch.object(members[0].model, "predict_proba", side_effect=RuntimeError("x")):
        ll = member_log_likelihoods(members, frame, ["close"])
        assert members[0].name in ll
    # transition fallbacks
    tm = collect_transition([], 3)
    assert tm.shape == (3, 3)
    assert abs(tm.sum(axis=1) - 1).max() < 1e-8
    # mismatched state_map
    m0 = members[0]
    m0.state_map = np.ones((5, 3))
    m0.metadata["fitted"] = True
    _ = collect_transition([m0], 3)


@pytest.mark.unit
def test_model_branches_and_ssm_smooth() -> None:
    frame = pl.DataFrame({"open_time": list(range(40)), "close": np.linspace(1, 2, 40)})
    # feature_columns from settings
    model = EnsembleRegimeModel(settings=_settings())
    model.fit(frame)  # uses columns.feature_columns
    model.partial_fit(frame.slice(30, 10))
    # resolve names edge
    from iqrp.app.regimes.ensemble.model import _default_names

    assert len(_default_names(4, ("a",))) == 4
    assert len(_default_names(2, ("a", "b", "c"))) == 2
    # calibrator short series
    Calibrator(method="temperature").fit(np.ones((3, 2)) / 2, np.array([0, 1, 0]))
    # ECE empty bins
    assert expected_calibration_error(np.array([[0.9, 0.1]] * 5), np.zeros(5, dtype=int)) >= 0
    # platt fail path
    with patch("scipy.optimize.minimize", return_value=MagicMock(success=False, x=np.array([1.0]))):
        Calibrator(method="temperature").fit(np.eye(5, 3) + 0.1, np.arange(5) % 3)
        Calibrator(method="platt").fit(np.eye(5, 3) + 0.1, np.arange(5) % 3)
    # disagreement single member
    assert pairwise_disagreement([np.eye(3)]).shape[0] == 3
    # evaluator drawdown branch with uniform conf
    ev = EnsembleEvaluator().evaluate_member(
        proba=np.ones((10, 3)) / 3,
        hard=np.zeros(10, dtype=int),
        truth=np.zeros(10, dtype=int),
    )
    assert "drawdown_accuracy" in ev
    # confidence edge cases
    assert expected_persistence(np.array([1.0]), 0) == 1.0
    assert forecast_uncertainty(np.array([0.5, 0.5])).shape[0] == 1
    assert posterior_confidence(np.array([[0.2, 0.8]])).shape[0] == 1
    # SSM smooth / filter paths
    ssm = EnsembleStateSpaceModel(settings=_settings())
    ssm.fit(frame, observation_columns=["close"])
    sm = ssm.smooth(frame, observation_columns=["close"], lag=1)
    assert sm.smoothed_states.shape[0] == 40
    # serializer without sidecar
    path = Path("/tmp/ens_noside.json")
    EnsembleSerializer().save(model, path)
    path.with_suffix(".npz").unlink(missing_ok=True)
    loaded = EnsembleSerializer().load(path, model_cls=EnsembleRegimeModel)
    assert loaded.is_fitted


@pytest.mark.unit
def test_weighting_registry_viz_gaps(tmp_path: Path) -> None:
    assert user_weights(None, ["a", "b"]).sum() == pytest.approx(1)
    assert stability_weights({"a": np.ones(2), "b": np.eye(2)}, ["a", "b"]).sum() == pytest.approx(1)
    # compute_weights branches
    assert compute_weights("calibration", names=["a"], ece_scores={"a": 0.2}).shape[0] == 1
    assert compute_weights("stability", names=["a"], proba_histories={"a": np.eye(2)}).shape[0] == 1
    assert compute_weights(
        "adaptive",
        names=["a", "b"],
        current=np.array([0.5, 0.5]),
        score_matrix=np.array([[0.2, 0.8]]),
    ).sum() == pytest.approx(1)
    # build_state_map leftover / unknown names
    m = build_state_map(("foo", "bar", "baz", "qux"), ("bull", "bear"))
    assert m.shape == (4, 2)
    # trainer all fail
    settings = _settings()
    members = EnsembleRegistry(settings).create_members()
    for m in members:
        m.model.fit = MagicMock(side_effect=RuntimeError("x"))  # type: ignore[method-assign]
    with pytest.raises(ValidationError):
        EnsembleTrainer(settings).fit(pl.DataFrame({"close": [1.0, 2.0, 3.0]}), ["close"], members=members)
    plot_agreement_heatmap(np.eye(1), ["a"], tmp_path / "h.svg")
    plot_probability_dashboard(np.array([0.5, 0.5]), ("a", "b"), tmp_path / "p.svg")
