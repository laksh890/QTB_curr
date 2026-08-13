"""Abstract regime model contract for all future algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.regimes.base.evaluator import EvaluationReport, RegimeEvaluator
from iqrp.app.regimes.base.forecast import RegimeForecast
from iqrp.app.regimes.base.persistence import PersistenceEngine
from iqrp.app.regimes.base.probabilities import ProbabilityEngine
from iqrp.app.regimes.base.regime import RegimeResult
from iqrp.app.regimes.base.state import RegimeState
from iqrp.app.regimes.base.transition import RegimeTransition


@dataclass(frozen=True, slots=True)
class RegimeModelMeta:
    name: str
    version: str
    description: str
    n_states: int
    algorithm_family: str
    parameters: dict[str, Any] = field(default_factory=dict)
    state_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "n_states": self.n_states,
            "algorithm_family": self.algorithm_family,
            "parameters": dict(self.parameters),
            "state_names": list(self.state_names),
        }


class RegimeModel(ABC):
    """Interchangeable regime detection algorithm interface.

    Downstream code must depend only on this contract — never on concrete
    Markov / HMM / GMM / neural implementations.
    """

    meta: RegimeModelMeta

    def __init__(self) -> None:
        self._fitted: bool = False
        self._transition_matrix: np.ndarray | None = None
        self._state_names: tuple[str, ...] = ()

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _require_fitted(self) -> None:
        if not self._fitted:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                f"Regime model '{self.meta.name}' is not fitted",
                code="REGIME_NOT_FITTED",
            )

    @abstractmethod
    def fit(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> RegimeModel:
        """Fit the regime model on ``frame``."""

    @abstractmethod
    def predict(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
        """Hard state sequence, shape (T,)."""

    @abstractmethod
    def predict_proba(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        """State probabilities, shape (T, K)."""

    def transition_matrix(self) -> np.ndarray:
        self._require_fitted()
        if self._transition_matrix is None:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                "Transition matrix unavailable",
                code="REGIME_NO_TRANSITION_MATRIX",
            )
        return np.asarray(self._transition_matrix, dtype=np.float64)

    def state_sequence(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> list[RegimeState]:
        """Typed state objects for each timestamp."""
        self._require_fitted()
        ids = self.predict(frame, feature_columns)
        proba = self.predict_proba(frame, feature_columns)
        names = self._state_names or tuple(f"state_{i}" for i in range(proba.shape[1]))
        tm = self.transition_matrix()
        persist_scores = PersistenceEngine.persistence_score(tm)
        ts_col = "open_time" if "open_time" in frame.columns else None
        features = tuple(feature_columns or [])
        states: list[RegimeState] = []
        for i, sid in enumerate(ids):
            sid_i = int(sid)
            p = float(proba[i, sid_i]) if proba.ndim == 2 else float(proba[sid_i])
            ts = frame[ts_col][i] if ts_col else None
            states.append(
                RegimeState(
                    state_id=sid_i,
                    state_name=names[sid_i] if sid_i < len(names) else f"state_{sid_i}",
                    probability=p,
                    confidence=p,
                    persistence=float(persist_scores.get(sid_i, 0.0)),
                    start_time=ts,
                    end_time=None,
                    duration=None,
                    features_used=features,
                    model_version=self.meta.version,
                    timestamp=ts,
                )
            )
        return states

    def forecast(self, frame: pl.DataFrame, steps: int = 1) -> RegimeForecast:
        """Forecast state distribution for 1..N steps."""
        self._require_fitted()
        proba = self.predict_proba(frame)
        current = proba[-1] if proba.ndim == 2 else proba
        tm = self.transition_matrix()
        future = ProbabilityEngine.forecast_probability(current, tm, steps)
        names = self._state_names or tuple(f"state_{i}" for i in range(len(current)))
        expected = PersistenceEngine.expected_duration_from_transition(tm)
        return RegimeForecast.from_probabilities(
            future, state_names=names, expected_duration=expected
        )

    def save(self, path: Path) -> Path:
        """Serialize model artifact (JSON metadata + arrays)."""
        from iqrp.app.regimes.services.serializer import RegimeSerializer

        self._require_fitted()
        return RegimeSerializer().save(self, path)

    @classmethod
    def load(cls, path: Path) -> RegimeModel:
        """Load model artifact into a fresh instance of ``cls``."""
        from iqrp.app.regimes.services.serializer import RegimeSerializer

        return RegimeSerializer().load(path, model_cls=cls)

    def evaluate(
        self,
        frame: pl.DataFrame,
        *,
        true_states: np.ndarray | None = None,
        feature_columns: list[str] | None = None,
    ) -> EvaluationReport:
        self._require_fitted()
        pred = self.predict(frame, feature_columns)
        proba = self.predict_proba(frame, feature_columns)
        tm = self.transition_matrix()
        return RegimeEvaluator().evaluate(
            predicted=pred,
            probabilities=proba,
            transition_matrix=tm,
            true_states=true_states,
        )

    def detect(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        forecast_steps: int = 5,
    ) -> RegimeResult:
        """Convenience: predict + enrich into :class:`RegimeResult`."""
        self._require_fitted()
        ids = self.predict(frame, feature_columns)
        proba = self.predict_proba(frame, feature_columns)
        tm = self.transition_matrix()
        states = self.state_sequence(frame, feature_columns)
        transitions = self._extract_transitions(ids, tm, frame)
        bundle = ProbabilityEngine.bundle(proba, tm, forecast_steps=forecast_steps)
        persistence = PersistenceEngine.analyze(ids, tm)
        fc = self.forecast(frame, steps=max(1, forecast_steps))
        ts_col = "open_time" if "open_time" in frame.columns else None
        timestamps = list(frame[ts_col].to_list()) if ts_col else [None] * len(ids)
        return RegimeResult(
            model_name=self.meta.name,
            model_version=self.meta.version,
            states=states,
            state_ids=ids,
            state_probabilities=proba,
            transition_matrix=tm,
            transitions=transitions,
            probabilities=bundle,
            persistence=persistence,
            forecast=fc,
            timestamps=timestamps,
            feature_columns=tuple(feature_columns or []),
        )

    def _extract_transitions(
        self, ids: np.ndarray, tm: np.ndarray, frame: pl.DataFrame
    ) -> list[RegimeTransition]:
        names = self._state_names or tuple(f"state_{i}" for i in range(tm.shape[0]))
        ts_col = "open_time" if "open_time" in frame.columns else None
        out: list[RegimeTransition] = []
        for i in range(1, len(ids)):
            prev, cur = int(ids[i - 1]), int(ids[i])
            if prev == cur:
                continue
            ts = frame[ts_col][i] if ts_col else None
            out.append(
                RegimeTransition(
                    previous_state=prev,
                    current_state=cur,
                    probability=float(tm[prev, cur]),
                    confidence=float(tm[prev, cur]),
                    timestamp=ts,
                    previous_name=names[prev] if prev < len(names) else None,
                    current_name=names[cur] if cur < len(names) else None,
                )
            )
        return out

    def get_params(self) -> dict[str, Any]:
        return dict(self.meta.parameters)

    def export_state(self) -> dict[str, Any]:
        """Serializable fitted state for save/load."""
        self._require_fitted()
        return {
            "meta": self.meta.to_dict(),
            "transition_matrix": (
                None
                if self._transition_matrix is None
                else np.asarray(self._transition_matrix).tolist()
            ),
            "state_names": list(self._state_names),
            "fitted": self._fitted,
            "algorithm_state": self._algorithm_state(),
        }

    def import_state(self, payload: dict[str, Any]) -> None:
        tm = payload.get("transition_matrix")
        self._transition_matrix = None if tm is None else np.asarray(tm, dtype=np.float64)
        self._state_names = tuple(payload.get("state_names") or ())
        self._fitted = bool(payload.get("fitted", False))
        self._load_algorithm_state(payload.get("algorithm_state") or {})

    @abstractmethod
    def _algorithm_state(self) -> dict[str, Any]:
        """Algorithm-specific fitted parameters."""

    @abstractmethod
    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        """Restore algorithm-specific fitted parameters."""
