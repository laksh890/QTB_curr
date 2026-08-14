"""Core unit tests for ensemble regime engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.base.regime_model import RegimeModel, RegimeModelMeta
from iqrp.app.regimes.base.registry import get_registry, register_regime_model
from iqrp.app.regimes.ensemble import (
    Calibrator,
    EnsembleRegimeModel,
    EnsembleSettings,
    EnsembleStateSpaceModel,
    combine,
    compute_weights,
    expected_calibration_error,
)
from iqrp.app.regimes.ensemble.disagreement import disagreement_report
from iqrp.app.regimes.ensemble.visualization import (
    plot_agreement_heatmap,
    plot_confidence_timeline,
    plot_probability_dashboard,
    plot_regime_timeline,
    plot_weight_evolution,
)


@register_regime_model
class _StubRegimeA(RegimeModel):
    meta = RegimeModelMeta(
        name="stub_a",
        version="1.0.0",
        description="stub",
        n_states=3,
        algorithm_family="stub",
        state_names=("bear", "sideways", "bull"),
    )

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self._p: np.ndarray | None = None
        self._state_names = self.meta.state_names

    def fit(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> _StubRegimeA:
        n = frame.height
        # trend: early bear, mid sideways, late bull
        p = np.zeros((n, 3))
        for i in range(n):
            if i < n // 3:
                p[i] = [0.7, 0.2, 0.1]
            elif i < 2 * n // 3:
                p[i] = [0.2, 0.6, 0.2]
            else:
                p[i] = [0.1, 0.2, 0.7]
        self._p = p
        self._transition_matrix = np.array(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=np.float64
        )
        self._fitted = True
        return self

    def predict(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
        return np.argmax(self.predict_proba(frame, feature_columns), axis=1)

    def predict_proba(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        assert self._p is not None
        n = frame.height
        if self._p.shape[0] == n:
            return self._p.copy()
        return np.tile(self._p[-1], (n, 1))

    def _algorithm_state(self) -> dict:
        return {"p": None if self._p is None else self._p.tolist()}

    def _load_algorithm_state(self, state: dict) -> None:
        self._p = None if state.get("p") is None else np.asarray(state["p"], dtype=np.float64)
        self._fitted = self._p is not None
        self._transition_matrix = np.eye(3) * 0.8 + 0.1


@register_regime_model
class _StubRegimeB(RegimeModel):
    meta = RegimeModelMeta(
        name="stub_b",
        version="1.0.0",
        description="stub b",
        n_states=2,
        algorithm_family="stub",
        state_names=("bearish", "bullish"),
    )

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self._p: np.ndarray | None = None
        self._state_names = self.meta.state_names

    def fit(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> _StubRegimeB:
        n = frame.height
        cols = feature_columns or [c for c in frame.columns if frame[c].dtype.is_numeric()]
        primary = cols[0] if cols else frame.columns[0]
        series = frame[primary].to_numpy().astype(np.float64).reshape(-1)
        z = (series - series.mean()) / (series.std() + 1e-9)
        p_bull = 1.0 / (1.0 + np.exp(-z))
        self._p = np.column_stack([1 - p_bull, p_bull])
        self._transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
        self._fitted = True
        return self

    def predict(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
        return np.argmax(self.predict_proba(frame, feature_columns), axis=1)

    def predict_proba(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        n = frame.height
        if self._p is not None and self._p.shape[0] == n:
            return self._p.copy()
        return self.fit(frame, feature_columns)._p  # type: ignore[return-value]

    def _algorithm_state(self) -> dict:
        return {"p": None if self._p is None else self._p.tolist()}

    def _load_algorithm_state(self, state: dict) -> None:
        self._p = None if state.get("p") is None else np.asarray(state["p"], dtype=np.float64)
        self._fitted = self._p is not None
        self._transition_matrix = np.eye(2) * 0.9 + 0.05


def _settings(**kw: object) -> EnsembleSettings:
    data = {
        **EnsembleSettings.default().model_dump(),
        "n_states": 3,
        "state_names": ("bull", "bear", "sideways"),
        "discovery_modules": ("iqrp.app.regimes.models.mock",),
        "member_names": ("stub_a", "stub_b", "mock_regime"),
        "training": {"validation_fraction": 0.3, "min_members": 1},
        "calibration": {"enabled": True, "method": "temperature", "temperature": 1.0},
    }
    data.update(kw)
    return EnsembleSettings.from_mapping(data)


def _frame(n: int = 90, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    return pl.DataFrame(
        {"open_time": list(range(n)), "close": close, "f1": np.diff(close, prepend=close[0])}
    )


@pytest.mark.unit
def test_combine_and_weights() -> None:
    rng = np.random.default_rng(0)
    probs = [rng.dirichlet(np.ones(3), size=20) for _ in range(3)]
    w = np.array([0.5, 0.3, 0.2])
    for method in (
        "majority",
        "weighted",
        "soft_voting",
        "bma",
        "stacking",
        "confidence",
        "dynamic",
        "meta",
    ):
        out = combine(probs, w, method=method, n_states=3, log_evidence=np.log(w), scores=w)  # type: ignore[arg-type]
        assert out.shape == (20, 3)
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)
    names = ["a", "b", "c"]
    truth = np.argmax(probs[0], axis=1)
    preds = {n: np.argmax(p, axis=1) for n, p in zip(names, probs, strict=False)}
    assert compute_weights("equal", names=names).shape[0] == 3
    assert compute_weights(
        "accuracy", names=names, predictions=preds, truth=truth
    ).sum() == pytest.approx(1.0)
    assert compute_weights(
        "user", names=names, user={"a": 2, "b": 1, "c": 1}
    ).sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_calibration_methods() -> None:
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(3), size=80)
    y = np.argmax(p, axis=1)
    for method in ("temperature", "platt", "isotonic", "dirichlet", "none"):
        cal = Calibrator(method=method)  # type: ignore[arg-type]
        cal.fit(p, y)
        out = cal.transform(p)
        assert out.shape == p.shape
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)
    assert expected_calibration_error(p, y) >= 0


@pytest.mark.unit
def test_model_api(tmp_path: Path) -> None:
    # ensure stubs registered
    assert "stub_a" in get_registry().list_names()
    frame = _frame(100, 2)
    model = EnsembleRegimeModel(settings=_settings(), random_seed=2)
    model.fit(frame, feature_columns=["close", "f1"])
    pred = model.predict(frame, feature_columns=["close", "f1"])
    proba = model.predict_proba(frame, feature_columns=["close", "f1"])
    assert pred.shape[0] == frame.height
    assert proba.shape[1] == 3
    fc = model.forecast(frame, steps=4)
    assert fc.steps == 4
    assert "confidence" in model.confidence(frame)
    assert "mean_consensus" in model.consensus(frame)
    assert model.weights()
    board = model.leaderboard(frame)
    assert board[0]["rank"] == 1
    model.partial_fit(frame.slice(90, 10), feature_columns=["close", "f1"])
    truth = pred.copy()
    model.calibrate(frame, truth, feature_columns=["close", "f1"])
    diag = model.diagnostics(frame, true_states=truth)
    assert "leaderboard" in diag
    report = model.evaluate(frame, true_states=truth, feature_columns=["close", "f1"])
    assert "metrics" in report
    path = model.save(tmp_path / "ens.json")
    loaded = EnsembleRegimeModel.load(path)
    assert loaded.is_fitted
    plot_regime_timeline(pred, tmp_path / "tl.svg")
    plot_confidence_timeline(proba.max(axis=1), tmp_path / "cf.svg")
    plot_probability_dashboard(proba, model._state_names, tmp_path / "pd.svg")
    plot_weight_evolution([model.weights()], tmp_path / "we.svg")
    plot_agreement_heatmap(np.eye(2), ["a", "b"], tmp_path / "hm.svg")
    # state space adapter
    ssm = EnsembleStateSpaceModel(settings=_settings(), random_seed=3)
    ssm.fit(frame, observation_columns=["close", "f1"])
    assert ssm.predict_proba(frame, observation_columns=["close", "f1"]).shape[1] == 3
    assert ssm.forecast(frame, observation_columns=["close", "f1"], horizon=2).horizon == 2


@pytest.mark.unit
def test_disagreement_report() -> None:
    rng = np.random.default_rng(4)
    probs = [rng.dirichlet(np.ones(3), size=15) for _ in range(2)]
    rep = disagreement_report(probs, names=["a", "b"])
    assert 0 <= rep["mean_consensus"] <= 1
