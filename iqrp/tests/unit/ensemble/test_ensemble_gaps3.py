"""Final coverage push for ensemble package (>98%)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.base.regime_model import RegimeModel, RegimeModelMeta
from iqrp.app.regimes.ensemble.calibration import (
    Calibrator,
    _isotonic_regression,
    brier_score,
    expected_calibration_error,
)
from iqrp.app.regimes.ensemble.confidence import confidence_report
from iqrp.app.regimes.ensemble.config import EnsembleSettings
from iqrp.app.regimes.ensemble.evaluator import EnsembleEvaluator
from iqrp.app.regimes.ensemble.model import (
    EnsembleRegimeModel,
    EnsembleStateSpaceModel,
    _as_frame,
    _default_names,
)
from iqrp.app.regimes.ensemble.orchestrator import (
    collect_transition,
    member_log_likelihoods,
    predict_members,
)
from iqrp.app.regimes.ensemble.registry import (
    EnsembleMember,
    EnsembleRegistry,
    _default_map,
    build_state_map,
)
from iqrp.app.regimes.ensemble.serializer import _json_default
from iqrp.app.regimes.ensemble.trainer import EnsembleTrainer
from iqrp.app.regimes.ensemble.visualization import (
    plot_agreement_heatmap,
    plot_member_timelines,
    plot_regime_timeline,
)
from iqrp.app.regimes.ensemble.weighting import compute_weights, rolling_weights
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
        "calibration": {"enabled": True, "method": "none", "temperature": 1.0},
        "columns": {"timestamp": "open_time", "feature_columns": ("close",)},
        "visualization": {"enabled": True, "max_points": 500},
    }
    data.update(kw)
    return EnsembleSettings.from_mapping(data)


@pytest.mark.unit
def test_default_names_and_as_frame() -> None:
    assert _default_names(3, None) == ("regime_0", "regime_1", "regime_2")
    df = _as_frame(np.array([1.0, 2.0, 3.0]), None)
    assert "f0" in df.columns
    df2 = _as_frame(np.array([[1.0], [2.0]]), ["close"])
    assert df2.columns == ["close"]
    assert isinstance(_as_frame(pl.DataFrame({"x": [1]}), None), pl.DataFrame)


@pytest.mark.unit
def test_partial_fit_cold_and_member_partial() -> None:
    frame = pl.DataFrame({"open_time": list(range(30)), "close": np.linspace(1, 2, 30)})
    cold = EnsembleRegimeModel(settings=_settings(online={"warm_start": False, "weight_update": False}))
    cold.partial_fit(frame)  # routes to fit
    assert cold.is_fitted

    # unfitted -> fit via partial_fit
    m2 = EnsembleRegimeModel(settings=_settings())
    m2.partial_fit(frame)
    assert m2.is_fitted

    # member with partial_fit success + failure
    class WithPartial(_StubRegimeA):
        meta = RegimeModelMeta(
            name="stub_partial",
            version="1.0.0",
            description="stub",
            n_states=3,
            algorithm_family="stub",
            state_names=("bear", "sideways", "bull"),
        )

        def partial_fit(self, frame: pl.DataFrame, feature_columns: list[str] | None = None):
            return self.fit(frame, feature_columns)

    class BrokenPartial(_StubRegimeA):
        meta = RegimeModelMeta(
            name="stub_broken_pf",
            version="1.0.0",
            description="stub",
            n_states=3,
            algorithm_family="stub",
            state_names=("bear", "sideways", "bull"),
        )

        def partial_fit(self, frame: pl.DataFrame, feature_columns: list[str] | None = None):
            raise RuntimeError("nope")

    model = EnsembleRegimeModel(settings=_settings())
    model.fit(frame)
    model.members = [
        EnsembleMember(name="ok", model=WithPartial().fit(frame, ["close"]), metadata={"fitted": True}),
        EnsembleMember(name="bad", model=BrokenPartial().fit(frame, ["close"]), metadata={"fitted": True}),
    ]
    model._train_frame = None
    model.partial_fit(frame.slice(20, 10), ["close"])
    assert model._train_frame is not None


@pytest.mark.unit
def test_weights_mismatch_and_resolve_features() -> None:
    frame = pl.DataFrame({"open_time": list(range(25)), "close": np.linspace(1, 2, 25)})
    model = EnsembleRegimeModel(settings=_settings(columns={"timestamp": "open_time", "feature_columns": ()}))
    model.fit(frame, ["close"])
    model._weights = np.array([1.0])  # mismatch vs members
    w = model.weights()
    assert abs(sum(w.values()) - 1.0) < 1e-8
    assert model._resolve_features(frame) is None


@pytest.mark.unit
def test_import_state_edges() -> None:
    frame = pl.DataFrame({"open_time": list(range(30)), "close": np.linspace(1, 2, 30)})
    model = EnsembleRegimeModel(settings=_settings())
    model.fit(frame)
    state = model.export_state()
    algo = state["algorithm_state"]
    # inject unknown member + broken import
    algo["member_states"].append(
        {"name": "ghost", "weight": 0.1, "state_map": None, "model_state": None, "metadata": {}}
    )
    if algo["member_states"]:
        algo["member_states"][0]["model_state"] = {"broken": True}
        with patch.object(
            type(model.members[0].model),
            "import_state",
            side_effect=RuntimeError("x"),
        ):
            other = EnsembleRegimeModel(settings=_settings())
            other.import_state(state)
    with patch.object(EnsembleRegistry, "create_members", side_effect=RuntimeError("disc")):
        other2 = EnsembleRegimeModel(settings=_settings())
        other2.import_state(state)
        assert other2.members is not None


@pytest.mark.unit
def test_ssm_predict_forecast_1d_and_state() -> None:
    frame = pl.DataFrame({"open_time": list(range(30)), "close": np.linspace(1, 2, 30)})
    ssm = EnsembleStateSpaceModel(settings=_settings())
    ssm.fit(frame, observation_columns=["close"])
    pred = ssm.predict(frame, observation_columns=["close"])
    assert pred.shape[0] == 30
    # force 1d probabilities path in forecast
    with patch.object(
        ssm._engine,
        "forecast",
        return_value=MagicMock(
            probabilities=np.array([0.2, 0.5, 0.3]),
            expected_duration={0: 2.0},
        ),
    ):
        fc = ssm.forecast(frame, observation_columns=["close"], horizon=2)
        assert fc.horizon == 2
    st = ssm._algorithm_state()
    ssm2 = EnsembleStateSpaceModel(settings=_settings())
    ssm2._load_algorithm_state(st)
    assert ssm2.is_fitted
    # ndarray observations
    arr = np.linspace(1, 2, 30).reshape(-1, 1)
    _ = ssm.filter(arr, observation_columns=["close"])


@pytest.mark.unit
def test_orchestrator_predict_exception_and_skips() -> None:
    frame = pl.DataFrame({"close": np.linspace(1, 2, 20)})
    members = EnsembleRegistry(_settings()).create_members()
    for m in members:
        m.model.fit(frame, ["close"])
        m.metadata["fitted"] = True
    with patch.object(members[0].model, "predict_proba", side_effect=RuntimeError("x")):
        mapped, _, names = predict_members(members, frame, ["close"], n_canonical=3, parallel=True)
        assert members[0].name not in names
        assert mapped
    # unfitted skip in LL / transition
    members[0].metadata["fitted"] = False
    members[0].model._fitted = False
    ll = member_log_likelihoods(members, frame, ["close"])
    assert members[0].name not in ll or members[0].metadata.get("fitted")
    tm = collect_transition(members, 3)
    assert tm.shape == (3, 3)
    # transition exception continue
    members[1].metadata["fitted"] = True
    with patch.object(members[1].model, "transition_matrix", side_effect=RuntimeError("tm")):
        _ = collect_transition([members[1]], 3)


@pytest.mark.unit
def test_registry_map_and_create_kwargs() -> None:
    m = EnsembleMember(name="x", model=_StubRegimeA())
    m.state_map = None
    out = m.map_proba(np.array([0.2, 0.3, 0.5]), 3)
    assert out.shape == (1, 3)
    # k > n_canonical fold; k < n_canonical leftover (hits pass branch)
    folded = _default_map(np.ones((5, 5)) / 5, 3)
    assert folded.shape == (5, 3)
    small = _default_map(np.ones((4, 1)), 3)
    assert small.shape == (4, 3)
    sm = build_state_map(("unknown_a", "unknown_b"), ("bull", "bear", "sideways"))
    assert sm.shape == (2, 3)
    # TypeError on create kwargs
    with patch.object(EnsembleRegistry, "discover", return_value=["stub_a", "stub_b"]):
        with patch("iqrp.app.regimes.ensemble.registry.get_registry") as gr:
            reg = MagicMock()
            a, b = _StubRegimeA(), _StubRegimeB()
            reg.create.side_effect = [TypeError("kwargs"), a, TypeError("kwargs"), b]
            gr.return_value = reg
            members = EnsembleRegistry(_settings()).create_members()
            assert len(members) == 2


@pytest.mark.unit
def test_calibration_1d_isotonic_backtrack_brier() -> None:
    c = Calibrator(method="none")
    c.fit(np.array([0.1, 0.9, 0.5, 0.4, 0.6, 0.7]), np.array([0, 1, 0, 0, 1, 1]))
    out = c.transform(np.array([0.2, 0.8, 0.5]))
    assert out.ndim == 2
    assert expected_calibration_error(np.array([0.9, 0.1, 0.8]), np.array([0, 1, 0])) >= 0
    assert brier_score(np.array([0.9, 0.1, 0.8]), np.array([0, 1, 0])) >= 0
    # force PAVA backtrack: decreasing sequence that merges then backtracks
    y = np.array([0.9, 0.8, 0.7, 0.1, 0.05, 0.9])
    iso = _isotonic_regression(y)
    assert iso.shape == y.shape
    assert np.all(np.diff(iso) >= -1e-12)


@pytest.mark.unit
def test_confidence_1d_evaluator_drawdown_else() -> None:
    rep = confidence_report(np.array([0.1, 0.2, 0.7]), [], transition=np.eye(3))
    assert "confidence" in rep
    # empty drawdown mask via patch
    with patch("iqrp.app.regimes.ensemble.evaluator.np.any", return_value=False):
        m = EnsembleEvaluator().evaluate_member(
            proba=np.eye(4, 3) + 0.01,
            hard=np.array([0, 1, 2, 0]),
            truth=np.array([0, 1, 2, 0]),
        )
        assert m["drawdown_accuracy"] == m["accuracy"]


@pytest.mark.unit
def test_weighting_rolling_recent_fallback() -> None:
    assert rolling_weights(np.array([])).shape[0] == 1
    assert rolling_weights(np.ones(3)).shape[0] == 1
    w = compute_weights(
        "recent_accuracy",
        names=["a", "b"],
        predictions={"a": np.array([0, 1, 0]), "b": np.array([0, 0, 1])},
        truth=np.array([0, 1, 1]),
    )
    assert w.sum() == pytest.approx(1)
    w2 = compute_weights("rolling", names=["a", "b"], score_matrix=np.array([[0.1, 0.9], [0.4, 0.6]]))
    assert w2.sum() == pytest.approx(1)
    # fallback equal
    assert compute_weights("accuracy", names=["a", "b"]).sum() == pytest.approx(1)


@pytest.mark.unit
def test_trainer_with_truth_serializer_json_viz(tmp_path: Path) -> None:
    frame = pl.DataFrame({"close": np.linspace(1, 2, 40)})
    truth = np.array([0, 1, 2] * 13 + [0])
    settings = _settings()
    members = EnsembleRegistry(settings).create_members()
    result = EnsembleTrainer(settings).fit(frame, ["close"], true_states=truth[:40], members=members)
    assert result.ensemble_proba.shape[0] > 0
    assert _json_default(np.float64(1.5)) == 1.5
    assert _json_default(np.int64(3)) == 3
    assert _json_default(np.bool_(True)) is True
    with pytest.raises(TypeError):
        _json_default(object())
    # viz disabled + empty series
    off = _settings(visualization={"enabled": False, "max_points": 10})
    plot_agreement_heatmap(np.eye(2), ["a", "b"], tmp_path / "off.svg", settings=off)
    plot_regime_timeline(np.array([]), tmp_path / "empty.svg", settings=_settings())
    plot_member_timelines({"a": np.array([]), "b": np.array([1.0])}, tmp_path / "mix.svg")
