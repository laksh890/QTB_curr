"""Abstract state-space model contract for all future latent-state algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.base.forecast_result import ForecastResult
from iqrp.app.state_space.base.latent_state import LatentState
from iqrp.app.state_space.base.smoother_result import SmootherResult
from iqrp.app.state_space.config import StateSpaceSettings


@dataclass(frozen=True, slots=True)
class StateSpaceModelMeta:
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


class StateSpaceModel(ABC):
    """Interchangeable latent-state model interface.

    Downstream code must depend only on this contract — never on concrete
    Markov / HMM / Kalman / particle / DLM implementations.
    """

    meta: StateSpaceModelMeta

    def __init__(self) -> None:
        self._fitted: bool = False
        self._state_names: tuple[str, ...] = ()
        self._settings: StateSpaceSettings = StateSpaceSettings.default()

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _require_fitted(self) -> None:
        if not self._fitted:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                f"State-space model '{self.meta.name}' is not fitted",
                code="SS_NOT_FITTED",
            )

    @abstractmethod
    def fit(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> StateSpaceModel:
        """Fit model parameters on observation data."""

    @abstractmethod
    def filter(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> FilterResult:
        """Online / forward filtering pass."""

    @abstractmethod
    def smooth(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
        lag: int | None = None,
    ) -> SmootherResult:
        """Offline or fixed-lag smoothing pass."""

    @abstractmethod
    def predict(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        """Hard latent-state sequence, shape ``(T,)``."""

    @abstractmethod
    def predict_proba(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        """Filtered (or default) state probabilities, shape ``(T, K)``."""

    @abstractmethod
    def forecast(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        horizon: int | None = None,
        observation_columns: list[str] | None = None,
    ) -> ForecastResult:
        """Multi-step latent-state forecast."""

    @abstractmethod
    def sample(
        self,
        n_steps: int,
        *,
        initial_state: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate ``(states, observations)`` of length ``n_steps``."""

    @abstractmethod
    def log_likelihood(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        """Marginal log-likelihood of the observation sequence."""

    def state_sequence(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> list[LatentState]:
        self._require_fitted()
        ids = self.predict(observations, observation_columns=observation_columns)
        proba = self.predict_proba(observations, observation_columns=observation_columns)
        names = self._state_names or tuple(f"state_{i}" for i in range(proba.shape[1]))
        timestamps = _timestamps(observations, self._settings.columns.timestamp)
        states: list[LatentState] = []
        for i, sid in enumerate(ids):
            sid_i = int(sid)
            p = float(proba[i, sid_i])
            states.append(
                LatentState(
                    state_id=sid_i,
                    state_name=names[sid_i] if sid_i < len(names) else f"state_{sid_i}",
                    probability=p,
                    confidence=p,
                    timestamp=timestamps[i] if timestamps else None,
                    metadata={"model_version": self.meta.version},
                )
            )
        return states

    def save(self, path: Path) -> Path:
        from iqrp.app.state_space.storage.serializer import StateSpaceSerializer

        self._require_fitted()
        return StateSpaceSerializer().save(self, path)

    @classmethod
    def load(cls, path: Path) -> StateSpaceModel:
        from iqrp.app.state_space.storage.serializer import StateSpaceSerializer

        return StateSpaceSerializer().load(path, model_cls=cls)

    def evaluate(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        true_states: np.ndarray | None = None,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        from iqrp.app.state_space.evaluation.metrics import EvaluationMetrics

        self._require_fitted()
        pred = self.predict(observations, observation_columns=observation_columns)
        proba = self.predict_proba(observations, observation_columns=observation_columns)
        ll = self.log_likelihood(observations, observation_columns=observation_columns)
        return EvaluationMetrics().evaluate(
            predicted=pred,
            probabilities=proba,
            log_likelihood=ll,
            n_params=self._n_params(),
            n_samples=len(pred),
            true_states=true_states,
            transition_matrix=self._transition_matrix_or_none(),
        )

    def export_state(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "meta": self.meta.to_dict(),
            "state_names": list(self._state_names),
            "fitted": self._fitted,
            "algorithm_state": self._algorithm_state(),
        }

    def import_state(self, payload: dict[str, Any]) -> None:
        self._state_names = tuple(payload.get("state_names") or ())
        self._fitted = bool(payload.get("fitted", False))
        self._load_algorithm_state(payload.get("algorithm_state") or {})

    def _n_params(self) -> int:
        return int(self.meta.parameters.get("n_params", self.meta.n_states))

    def _transition_matrix_or_none(self) -> np.ndarray | None:
        return None

    @abstractmethod
    def _algorithm_state(self) -> dict[str, Any]:
        """Algorithm-specific fitted parameters."""

    @abstractmethod
    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        """Restore algorithm-specific fitted parameters."""


def _timestamps(observations: pl.DataFrame | np.ndarray, column: str) -> list[Any] | None:
    if isinstance(observations, pl.DataFrame) and column in observations.columns:
        return list(observations[column].to_list())
    return None
