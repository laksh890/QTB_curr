"""Institutional Hidden Markov Model (State Space + Regime adapters)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.math.probability.likelihood import aic as aic_score, bic as bic_score
from iqrp.app.regimes.base.forecast import RegimeForecast
from iqrp.app.regimes.base.regime_model import RegimeModel, RegimeModelMeta
from iqrp.app.regimes.base.registry import register_regime_model
from iqrp.app.regimes.hmm.baum_welch import baum_welch, em_step
from iqrp.app.regimes.hmm.config import HMMSettings
from iqrp.app.regimes.hmm.diagnostics import HMMDiagnostics
from iqrp.app.regimes.hmm.emissions import EmissionModel, emission_from_dict
from iqrp.app.regimes.hmm.evaluator import HMMEvaluator
from iqrp.app.regimes.hmm.forward import forward
from iqrp.app.regimes.hmm.prediction import current_state_distribution, forecast_states
from iqrp.app.regimes.hmm.smoothing import smooth as hmm_smooth
from iqrp.app.regimes.hmm.trainer import HMMTrainer, _n_params
from iqrp.app.regimes.hmm.transitions import HMMTransitions
from iqrp.app.regimes.hmm.viterbi import viterbi
from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.base.forecast_result import ForecastResult
from iqrp.app.state_space.base.registry import register_state_space_model
from iqrp.app.state_space.base.smoother_result import SmootherResult
from iqrp.app.state_space.base.state_space_model import StateSpaceModel, StateSpaceModelMeta


def _resolve_names(n_states: int, names: tuple[str, ...] | None) -> tuple[str, ...]:
    if names and len(names) >= n_states:
        return tuple(names[:n_states])
    if names:
        return tuple(names) + tuple(f"state_{i}" for i in range(len(names), n_states))
    return tuple(f"state_{i}" for i in range(n_states))


@register_state_space_model
class HiddenMarkovModel(StateSpaceModel):
    """Production HMM for latent regime detection (Gaussian / discrete emissions)."""

    meta = StateSpaceModelMeta(
        name="hmm",
        version="1.0.0",
        description="Hidden Markov Model for market regime detection",
        n_states=3,
        algorithm_family="hmm",
        parameters={},
        state_names=("state_0", "state_1", "state_2"),
    )

    def __init__(
        self,
        *,
        n_states: int | None = None,
        n_features: int | None = None,
        state_names: tuple[str, ...] | None = None,
        settings: HMMSettings | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self._hmm_settings = settings or HMMSettings.default()
        k = int(n_states if n_states is not None else self._hmm_settings.n_states)
        d = int(n_features if n_features is not None else self._hmm_settings.n_features)
        names = state_names if state_names is not None else self._hmm_settings.state_names
        self._state_names = _resolve_names(k, names)
        self._rng = np.random.default_rng(
            random_seed if random_seed is not None else self._hmm_settings.random_seed
        )
        self.transitions: HMMTransitions | None = None
        self.emissions: EmissionModel | None = None
        self._history: list[float] = []
        self._n_iter = 0
        self._converged = False
        self._train_obs: np.ndarray | None = None
        self._online_buffer: list[np.ndarray] = []
        self._update_counter = 0
        self.meta = StateSpaceModelMeta(
            name="hmm",
            version="1.0.0",
            description=self.meta.description,
            n_states=k,
            algorithm_family="hmm",
            parameters={
                "n_states": k,
                "n_features": d,
                "emission": self._hmm_settings.emission.type,
                "covariance_type": self._hmm_settings.emission.covariance_type,
            },
            state_names=self._state_names,
        )

    @property
    def state_names(self) -> tuple[str, ...]:
        return self._state_names

    def fit(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> HiddenMarkovModel:
        y = self._extract_obs(observations, observation_columns)
        s = self._hmm_settings
        emis_type = "gaussian" if s.emission.type == "multivariate_gaussian" else s.emission.type
        result = baum_welch(
            y,
            self.meta.n_states,
            emission_type=emis_type,
            covariance_type=s.emission.covariance_type,
            method=s.initialization.method,
            max_iter=s.training.max_iter,
            tol=s.training.tol,
            early_stopping=s.training.early_stopping,
            min_covar=s.training.min_covar,
            dirichlet_alpha=s.initialization.dirichlet_alpha,
            n_restarts=s.initialization.n_restarts,
            n_jobs=s.training.n_jobs,
            rng=self._rng,
        )
        self.transitions = result.transitions
        self.emissions = result.emissions
        self._history = list(result.history)
        self._n_iter = result.n_iter
        self._converged = result.converged
        self._train_obs = y
        self._fitted = True
        self._online_buffer = []
        return self

    def partial_fit(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> HiddenMarkovModel:
        y = self._extract_obs(observations, observation_columns)
        if not self._fitted or self.transitions is None or self.emissions is None:
            return self.fit(observations, observation_columns=observation_columns)
        s = self._hmm_settings
        if not s.online.warm_start:
            return self.fit(observations, observation_columns=observation_columns)
        self._online_buffer.append(y)
        self._update_counter += 1
        freq = max(1, int(s.online.update_frequency))
        if self._update_counter % freq != 0:
            return self
        chunk = (
            np.vstack(self._online_buffer)
            if len(self._online_buffer) > 1
            else self._online_buffer[0]
        )
        window = int(s.online.window_size)
        if window > 0 and self._train_obs is not None:
            hist = np.vstack([self._train_obs, chunk])
            chunk = hist[-window:]
        ll = em_step(chunk, self.transitions, self.emissions, min_covar=s.training.min_covar)
        self._history.append(ll)
        self._n_iter += 1
        self._train_obs = chunk if self._train_obs is None else np.vstack([self._train_obs, chunk])
        self._online_buffer = []
        return self

    def filter(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> FilterResult:
        self._require_fitted()
        assert self.transitions is not None and self.emissions is not None
        y = self._extract_obs(observations, observation_columns)
        log_e = self.emissions.log_prob(y)
        alpha, scales, ll = forward(
            log_e, self.transitions.transition, initial=self.transitions.initial
        )
        states = np.argmax(alpha, axis=1).astype(np.int64)
        return FilterResult(
            filtered_states=states,
            filtered_probabilities=alpha,
            log_likelihood=ll,
            normalization_constants=scales,
            metadata={"model": "hmm"},
        )

    def smooth(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
        lag: int | None = None,
    ) -> SmootherResult:
        del lag
        self._require_fitted()
        assert self.transitions is not None and self.emissions is not None
        y = self._extract_obs(observations, observation_columns)
        log_e = self.emissions.log_prob(y)
        result, _ = hmm_smooth(log_e, self.transitions.transition, initial=self.transitions.initial)
        return result

    def predict(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        return self.decode(observations, observation_columns=observation_columns)

    def predict_proba(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        return self.smooth(
            observations, observation_columns=observation_columns
        ).smoothed_probabilities

    def decode(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        self._require_fitted()
        assert self.transitions is not None and self.emissions is not None
        y = self._extract_obs(observations, observation_columns)
        log_e = self.emissions.log_prob(y)
        return viterbi(log_e, self.transitions.transition, initial=self.transitions.initial).states

    def forward(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        self._require_fitted()
        assert self.transitions is not None and self.emissions is not None
        y = self._extract_obs(observations, observation_columns)
        log_e = self.emissions.log_prob(y)
        return forward(log_e, self.transitions.transition, initial=self.transitions.initial)

    def backward(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        self._require_fitted()
        assert self.transitions is not None and self.emissions is not None
        y = self._extract_obs(observations, observation_columns)
        log_e = self.emissions.log_prob(y)
        _, scales, _ = forward(log_e, self.transitions.transition, initial=self.transitions.initial)
        from iqrp.app.regimes.hmm.backward import backward as hmm_backward

        return hmm_backward(log_e, self.transitions.transition, scales=scales)

    def forecast(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        horizon: int | None = None,
        observation_columns: list[str] | None = None,
    ) -> ForecastResult:
        self._require_fitted()
        assert self.transitions is not None
        proba = self.predict_proba(observations, observation_columns=observation_columns)
        default_h = self._hmm_settings.forecasting.default_horizon
        h = int(horizon if horizon is not None else default_h)
        return forecast_states(
            current_state_distribution(proba),
            self.transitions,
            horizon=h,
            state_names=self._state_names,
            confidence_level=self._hmm_settings.forecasting.confidence_level,
        )

    def sample(
        self,
        n_steps: int,
        *,
        initial_state: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        assert self.transitions is not None and self.emissions is not None
        rng = rng or self._rng
        from iqrp.app.math.stochastic.markov_utils import simulate_markov

        states = simulate_markov(
            self.transitions.transition, n_steps, initial=initial_state, rng=rng
        )
        obs = self.emissions.sample(states, rng=rng)
        return states, obs

    def log_likelihood(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        self._require_fitted()
        assert self.transitions is not None and self.emissions is not None
        y = self._extract_obs(observations, observation_columns)
        log_e = self.emissions.log_prob(y)
        _, _, ll = forward(log_e, self.transitions.transition, initial=self.transitions.initial)
        return float(ll)

    def aic(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        ll = self.log_likelihood(observations, observation_columns=observation_columns)
        assert self.emissions is not None
        return aic_score(-ll, _n_params(self.meta.n_states, self.emissions))

    def bic(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        y = self._extract_obs(observations, observation_columns)
        ll = self.log_likelihood(y)
        assert self.emissions is not None
        return bic_score(-ll, _n_params(self.meta.n_states, self.emissions), max(len(y), 1))

    def transition_matrix(self) -> np.ndarray:
        self._require_fitted()
        assert self.transitions is not None
        return self.transitions.transition.copy()

    def save(self, path: Path) -> Path:
        from iqrp.app.regimes.hmm.serializer import HMMSerializer

        self._require_fitted()
        return HMMSerializer().save(self, path)

    @classmethod
    def load(cls, path: Path) -> HiddenMarkovModel:
        from iqrp.app.regimes.hmm.serializer import HMMSerializer

        return HMMSerializer().load(path, model_cls=cls)

    def evaluate(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        true_states: np.ndarray | None = None,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        assert self.emissions is not None
        y = self._extract_obs(observations, observation_columns)
        pred = self.predict(y)
        proba = self.predict_proba(y)
        ll = self.log_likelihood(y)
        return HMMEvaluator().evaluate(
            true_states=true_states,
            predicted_states=pred,
            probabilities=proba,
            log_likelihood=ll,
            emissions=self.emissions,
            n_samples=len(y),
        )

    def diagnostics(
        self,
        observations: pl.DataFrame | np.ndarray | None = None,
        *,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        assert self.transitions is not None and self.emissions is not None
        if observations is None:
            if self._train_obs is None:
                from iqrp.app.core.exceptions import ValidationError

                raise ValidationError("No observations for diagnostics", code="HMM_NO_OBS")
            y = self._train_obs
        else:
            y = self._extract_obs(observations, observation_columns)
        proba = self.predict_proba(y)
        states = self.predict(y)
        return HMMDiagnostics().generate(
            states=states,
            probabilities=proba,
            transition=self.transitions.transition,
            emissions=self.emissions,
            history=self._history,
            converged=self._converged,
            n_iter=self._n_iter,
            state_names=self._state_names,
        )

    def select_model(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        y = self._extract_obs(observations, observation_columns)
        return HMMTrainer(self._hmm_settings).select_n_states(y, rng=self._rng)

    def _extract_obs(
        self,
        observations: pl.DataFrame | np.ndarray,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        if isinstance(observations, np.ndarray):
            y = np.asarray(observations)
            if self._hmm_settings.emission.type == "discrete":
                return y.reshape(-1)
            return y.reshape(-1, 1) if y.ndim == 1 else y
        cols = observation_columns
        if not cols:
            if self._hmm_settings.columns.observation_columns:
                cols = list(self._hmm_settings.columns.observation_columns)
            else:
                exclude = {
                    self._hmm_settings.columns.timestamp,
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "symbol",
                    "exchange",
                    "timeframe",
                    "state_id",
                }
                cols = [
                    c
                    for c, dt in zip(observations.columns, observations.dtypes, strict=False)
                    if c not in exclude and dt.is_numeric()
                ]
                if not cols and "close" in observations.columns:
                    cols = ["close"]
        if not cols:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("No observation columns", code="HMM_NO_OBS_COLS")
        y = observations.select(cols).to_numpy()
        if self._hmm_settings.emission.type == "discrete":
            return y.reshape(-1).astype(np.int64)
        return y.astype(np.float64)

    def _n_params(self) -> int:
        if self.emissions is None:
            return self.meta.n_states
        return _n_params(self.meta.n_states, self.emissions)

    def _transition_matrix_or_none(self) -> np.ndarray | None:
        if self.transitions is None:
            return None
        return self.transitions.transition.copy()

    def _algorithm_state(self) -> dict[str, Any]:
        assert self.transitions is not None and self.emissions is not None
        return {
            "transitions": self.transitions.to_dict(),
            "emissions": self.emissions.to_dict(),
            "transition": self.transitions.transition.tolist(),
            "initial": self.transitions.initial.tolist(),
            "history": list(self._history),
            "n_iter": self._n_iter,
            "converged": self._converged,
            "state_names": list(self._state_names),
            "settings": self._hmm_settings.model_dump(),
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        if "transitions" in state:
            self.transitions = HMMTransitions.from_dict(state["transitions"])
        if "emissions" in state:
            self.emissions = emission_from_dict(state["emissions"])
        self._history = list(state.get("history") or [])
        self._n_iter = int(state.get("n_iter", 0))
        self._converged = bool(state.get("converged", False))
        names = state.get("state_names")
        if names:
            self._state_names = tuple(names)
        k = self.transitions.n_states if self.transitions else self.meta.n_states
        self.meta = StateSpaceModelMeta(
            name=self.meta.name,
            version=self.meta.version,
            description=self.meta.description,
            n_states=k,
            algorithm_family="hmm",
            parameters={
                "n_states": k,
                "emission": (
                    self.emissions.to_dict().get("type") if self.emissions else "gaussian"
                ),
            },
            state_names=self._state_names,
        )


@register_regime_model
class HMMRegimeModel(RegimeModel):
    """RegimeModel adapter over :class:`HiddenMarkovModel`."""

    meta = RegimeModelMeta(
        name="hmm",
        version="1.0.0",
        description="Hidden Markov Model regime detector",
        n_states=3,
        algorithm_family="hmm",
        parameters={},
        state_names=("state_0", "state_1", "state_2"),
    )

    def __init__(
        self,
        *,
        n_states: int | None = None,
        settings: HMMSettings | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self._engine = HiddenMarkovModel(
            n_states=n_states, settings=settings, random_seed=random_seed
        )
        self.meta = RegimeModelMeta(
            name="hmm",
            version="1.0.0",
            description=self.meta.description,
            n_states=self._engine.meta.n_states,
            algorithm_family="hmm",
            parameters=dict(self._engine.meta.parameters),
            state_names=self._engine.state_names,
        )
        self._state_names = self._engine.state_names

    def fit(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> HMMRegimeModel:
        self._engine.fit(frame, observation_columns=feature_columns)
        self._fitted = True
        self._transition_matrix = self._engine.transition_matrix()
        self._state_names = self._engine.state_names
        return self

    def predict(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
        self._require_fitted()
        return self._engine.predict(frame, observation_columns=feature_columns)

    def predict_proba(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        return self._engine.predict_proba(frame, observation_columns=feature_columns)

    def forecast(self, frame: pl.DataFrame, steps: int = 1) -> RegimeForecast:
        self._require_fitted()
        fc = self._engine.forecast(frame, horizon=steps)
        return RegimeForecast.from_probabilities(
            (
                fc.step_distributions
                if fc.step_distributions is not None
                else fc.probability_distribution
            ),
            state_names=self._state_names,
            expected_duration=fc.expected_duration,
        )

    def _algorithm_state(self) -> dict[str, Any]:
        return self._engine.export_state()

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        self._engine.import_state(state)
        self._fitted = self._engine.is_fitted
        if self._engine.is_fitted:
            self._transition_matrix = self._engine.transition_matrix()
            self._state_names = self._engine.state_names
