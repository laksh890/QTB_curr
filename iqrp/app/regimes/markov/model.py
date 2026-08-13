"""Institutional discrete-time Markov Chain model (State Space + Regime adapters)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.math.stochastic.markov_utils import simulate_markov
from iqrp.app.regimes.base.forecast import RegimeForecast
from iqrp.app.regimes.base.regime_model import RegimeModel, RegimeModelMeta
from iqrp.app.regimes.base.registry import register_regime_model
from iqrp.app.regimes.markov.config import MarkovSettings
from iqrp.app.regimes.markov.diagnostics import MarkovDiagnostics
from iqrp.app.regimes.markov.estimator import TransitionEstimator
from iqrp.app.regimes.markov.evaluator import MarkovEvaluator
from iqrp.app.regimes.markov.forecast import MarkovForecaster
from iqrp.app.regimes.markov.persistence import PersistenceAnalyzer, expected_duration
from iqrp.app.regimes.markov.state_mapper import LabelStateMapper, StateMapper
from iqrp.app.regimes.markov.stationary import StationaryAnalyzer
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
class MarkovChainModel(StateSpaceModel):
    """First-order time-homogeneous discrete Markov chain.

    Observations are discrete state labels (fully observed). Integrates with the
    State Space Framework; use :class:`MarkovRegimeModel` for the Regime ABC.
    """

    meta = StateSpaceModelMeta(
        name="markov_chain",
        version="1.0.0",
        description="Discrete-time first-order Markov chain for regime modeling",
        n_states=3,
        algorithm_family="markov",
        parameters={},
        state_names=("state_0", "state_1", "state_2"),
    )

    def __init__(
        self,
        *,
        n_states: int | None = None,
        state_names: tuple[str, ...] | None = None,
        settings: MarkovSettings | None = None,
        state_mapper: LabelStateMapper | StateMapper | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self._markov_settings = settings or MarkovSettings.default()
        k = int(n_states if n_states is not None else self._markov_settings.n_states)
        names = state_names if state_names is not None else self._markov_settings.state_names
        self._state_names = _resolve_names(k, names)
        self._rng = np.random.default_rng(
            random_seed if random_seed is not None else self._markov_settings.random_seed
        )
        est = self._markov_settings.estimation
        self.estimator = TransitionEstimator(
            k,
            method=est.method,
            laplace_alpha=est.laplace_alpha,
            dirichlet_alpha=est.dirichlet_alpha,
            forgetting_factor=est.forgetting_factor,
        )
        if isinstance(state_mapper, LabelStateMapper):
            self._mapper = state_mapper
        elif callable(state_mapper):
            self._mapper = LabelStateMapper(n_states=k, custom_mapper=state_mapper)
        else:
            self._mapper = LabelStateMapper(n_states=k)
        self._forecaster = MarkovForecaster()
        self._stationary = StationaryAnalyzer()
        self._persistence = PersistenceAnalyzer()
        self._train_states: np.ndarray | None = None
        self._online_buffer: list[int] = []
        self._update_counter = 0
        self.meta = StateSpaceModelMeta(
            name="markov_chain",
            version="1.0.0",
            description=self.meta.description,
            n_states=k,
            algorithm_family="markov",
            parameters={
                "n_states": k,
                "method": est.method,
                "n_params": k * (k - 1),
            },
            state_names=self._state_names,
        )

    @property
    def state_names(self) -> tuple[str, ...]:
        return self._state_names

    @property
    def n_params(self) -> int:
        k = self.meta.n_states
        return int(k * (k - 1))

    def fit(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
        weights: Any | None = None,
    ) -> MarkovChainModel:
        states = self._extract_states(observations, observation_columns)
        self._mapper.fit(states)
        states = self._mapper.transform(states)
        w = self._extract_weights(observations, weights)
        window = int(self._markov_settings.estimation.window_size)
        if window > 0 and states.size > window:
            states = states[-window:]
            if w is not None and w.size >= states.size:
                w = w[-states.size :]
        self.estimator.fit(states, weights=w)
        self._train_states = states
        self._fitted = True
        self._online_buffer = []
        return self

    def partial_fit(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
        weights: Any | None = None,
    ) -> MarkovChainModel:
        if not self._fitted:
            return self.fit(observations, observation_columns=observation_columns, weights=weights)
        states = self._mapper.transform(self._extract_states(observations, observation_columns))
        w = self._extract_weights(observations, weights)
        freq = max(1, int(self._markov_settings.online.update_frequency))
        self._online_buffer.extend(int(x) for x in states.tolist())
        self._update_counter += 1
        if self._update_counter % freq == 0 and len(self._online_buffer) >= 2:
            buf = np.asarray(self._online_buffer, dtype=np.int64)
            self.estimator.partial_fit(buf, weights=w)
            window = int(self._markov_settings.estimation.window_size)
            if (
                window > 0
                and self._markov_settings.online.adaptive
                and self._train_states is not None
            ):
                hist = np.concatenate([self._train_states, buf])
                self.estimator.matrix.apply_window(
                    hist, window, alpha=self._markov_settings.estimation.laplace_alpha
                )
            self._train_states = (
                buf if self._train_states is None else np.concatenate([self._train_states, buf])
            )
            self._online_buffer = [int(buf[-1])]
        return self

    def filter(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> FilterResult:
        self._require_fitted()
        states = self._mapper.transform(self._extract_states(observations, observation_columns))
        proba = self._state_probabilities(states)
        scales = np.ones(len(states), dtype=np.float64)
        return FilterResult(
            filtered_states=states,
            filtered_probabilities=proba,
            log_likelihood=self.log_likelihood(
                observations, observation_columns=observation_columns
            ),
            normalization_constants=scales,
            metadata={"model": "markov_chain", "fully_observed": True},
        )

    def smooth(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
        lag: int | None = None,
    ) -> SmootherResult:
        del lag
        filt = self.filter(observations, observation_columns=observation_columns)
        return SmootherResult(
            smoothed_states=filt.filtered_states,
            smoothed_probabilities=filt.filtered_probabilities,
            backward_messages=np.ones_like(filt.filtered_probabilities),
            log_likelihood=filt.log_likelihood,
            metadata={"model": "markov_chain", "fully_observed": True},
        )

    def predict(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        self._require_fitted()
        return self._mapper.transform(self._extract_states(observations, observation_columns))

    def predict_proba(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        self._require_fitted()
        states = self._mapper.transform(self._extract_states(observations, observation_columns))
        return self._state_probabilities(states)

    def forecast(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        horizon: int | None = None,
        observation_columns: list[str] | None = None,
    ) -> ForecastResult:
        self._require_fitted()
        states = self._mapper.transform(self._extract_states(observations, observation_columns))
        default_h = self._markov_settings.forecasting.default_horizon
        h = int(horizon if horizon is not None else default_h)
        current = np.zeros(self.meta.n_states, dtype=np.float64)
        current[int(states[-1])] = 1.0
        return self._forecaster.forecast(
            current,
            self.transition_matrix(),
            horizon=h,
            state_names=self._state_names,
            confidence_level=self._markov_settings.forecasting.confidence_level,
        )

    def sample(
        self,
        n_steps: int,
        *,
        initial_state: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        rng = rng or self._rng
        states = simulate_markov(
            self.transition_matrix(),
            n_steps,
            initial=initial_state,
            rng=rng,
        )
        # Observations equal latent states for fully observed chain
        return states, states.astype(np.float64).reshape(-1, 1)

    def log_likelihood(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        self._require_fitted()
        states = self._mapper.transform(self._extract_states(observations, observation_columns))
        return float(self.estimator.log_likelihood(states))

    def transition_matrix(self) -> np.ndarray:
        self._require_fitted()
        return self.estimator.probability_matrix()

    def stationary_distribution(self) -> np.ndarray:
        self._require_fitted()
        return self._stationary.stationary_distribution(self.transition_matrix())

    def expected_duration(self) -> dict[int, float]:
        self._require_fitted()
        return expected_duration(self.transition_matrix())

    def state_probabilities(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        return self.predict_proba(observations, observation_columns=observation_columns)

    def save(self, path: Path) -> Path:
        from iqrp.app.regimes.markov.serializer import MarkovSerializer

        self._require_fitted()
        return MarkovSerializer().save(self, path)

    @classmethod
    def load(cls, path: Path) -> MarkovChainModel:
        from iqrp.app.regimes.markov.serializer import MarkovSerializer

        return MarkovSerializer().load(path, model_cls=cls)

    def evaluate(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        true_states: np.ndarray | None = None,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        states = self._mapper.transform(self._extract_states(observations, observation_columns))
        truth = true_states if true_states is not None else states
        pred = self.predict(observations, observation_columns=observation_columns)
        proba = self.predict_proba(observations, observation_columns=observation_columns)
        ll = self.log_likelihood(observations, observation_columns=observation_columns)
        # Forecast accuracy: 1-step ahead from t-1
        tm = self.transition_matrix()
        if states.size >= 2:
            fc_pred = np.array([int(np.argmax(tm[int(states[t])])) for t in range(states.size - 1)])
            fc_true = states[1:]
        else:
            fc_pred, fc_true = None, None
        return MarkovEvaluator().evaluate(
            true_states=truth,
            predicted_states=pred,
            probabilities=proba,
            transition=tm,
            log_likelihood=ll,
            n_params=self.n_params,
            forecast_true=fc_true,
            forecast_pred=fc_pred,
        )

    def diagnostics(
        self,
        observations: pl.DataFrame | np.ndarray | None = None,
        *,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        if observations is None:
            if self._train_states is None:
                from iqrp.app.core.exceptions import ValidationError

                raise ValidationError(
                    "No states available for diagnostics", code="MARKOV_NO_STATES"
                )
            states = self._train_states
        else:
            states = self._mapper.transform(self._extract_states(observations, observation_columns))
        return MarkovDiagnostics().generate(
            states=states,
            transition=self.transition_matrix(),
            counts=self.estimator.matrix.count_matrix(),
            min_count_warning=self._markov_settings.estimation.min_count_warning,
            state_names=self._state_names,
        )

    def persistence_report(
        self,
        observations: pl.DataFrame | np.ndarray | None = None,
        *,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        if observations is None:
            states = (
                self._train_states
                if self._train_states is not None
                else np.array([], dtype=np.int64)
            )
        else:
            states = self._mapper.transform(self._extract_states(observations, observation_columns))
        return self._persistence.analyze(
            states, self.transition_matrix(), n_states=self.meta.n_states
        )

    def stationary_analysis(self) -> dict[str, Any]:
        self._require_fitted()
        return self._stationary.analyze(self.transition_matrix())

    def _state_probabilities(self, states: np.ndarray) -> np.ndarray:
        """One-hot occupancy for observed states (fully observed chain)."""
        k = self.meta.n_states
        proba = np.zeros((states.size, k), dtype=np.float64)
        for t, s in enumerate(states):
            sid = int(s)
            if 0 <= sid < k:
                proba[t, sid] = 1.0
            else:
                proba[t] = 1.0 / k
        return proba

    def _extract_states(
        self,
        observations: pl.DataFrame | np.ndarray,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        if isinstance(observations, np.ndarray):
            return np.asarray(observations, dtype=np.int64).reshape(-1)
        col = None
        if observation_columns:
            col = observation_columns[0]
        elif self._markov_settings.columns.state_column in observations.columns:
            col = self._markov_settings.columns.state_column
        else:
            # Prefer integer-like columns named state*
            for c in observations.columns:
                if "state" in c.lower():
                    col = c
                    break
            if col is None:
                exclude = {
                    self._markov_settings.columns.timestamp,
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "symbol",
                    "exchange",
                    "timeframe",
                }
                for c, dt in zip(observations.columns, observations.dtypes, strict=False):
                    if c in exclude:
                        continue
                    if dt.is_integer() or dt.is_float():
                        col = c
                        break
        if col is None:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("No state column found", code="MARKOV_NO_STATE_COLUMN")
        return np.asarray(observations[col].to_list())

    def _extract_weights(
        self,
        observations: pl.DataFrame | np.ndarray,
        weights: Any | None,
    ) -> np.ndarray | None:
        if weights is not None:
            return np.asarray(weights, dtype=np.float64).reshape(-1)
        if isinstance(observations, pl.DataFrame):
            wc = self._markov_settings.columns.weight_column
            if wc and wc in observations.columns:
                return np.asarray(observations[wc].to_list(), dtype=np.float64)
        return None

    def _n_params(self) -> int:
        return self.n_params

    def _transition_matrix_or_none(self) -> np.ndarray | None:
        if not self._fitted:
            return None
        return self.transition_matrix()

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator.to_dict(),
            "transition_matrix": self.transition_matrix().tolist(),
            "counts": self.estimator.matrix.count_matrix().tolist(),
            "mapper": self._mapper.to_dict(),
            "state_names": list(self._state_names),
            "settings": self._markov_settings.model_dump(),
            "train_states": None if self._train_states is None else self._train_states.tolist(),
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        if "estimator" in state:
            self.estimator = TransitionEstimator.from_dict(state["estimator"])
        if "mapper" in state:
            self._mapper = LabelStateMapper.from_dict(state["mapper"])
        names = state.get("state_names")
        if names:
            self._state_names = tuple(names)
        ts = state.get("train_states")
        self._train_states = None if ts is None else np.asarray(ts, dtype=np.int64)
        k = self.estimator.n_states
        self.meta = StateSpaceModelMeta(
            name=self.meta.name,
            version=self.meta.version,
            description=self.meta.description,
            n_states=k,
            algorithm_family="markov",
            parameters={
                "n_states": k,
                "method": self.estimator.method,
                "n_params": k * (k - 1),
            },
            state_names=self._state_names,
        )


@register_regime_model
class MarkovRegimeModel(RegimeModel):
    """RegimeModel adapter over :class:`MarkovChainModel`."""

    meta = RegimeModelMeta(
        name="markov_chain",
        version="1.0.0",
        description="First-order Markov chain regime model",
        n_states=3,
        algorithm_family="markov",
        parameters={},
        state_names=("state_0", "state_1", "state_2"),
    )

    def __init__(
        self,
        *,
        n_states: int | None = None,
        state_names: tuple[str, ...] | None = None,
        settings: MarkovSettings | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self._engine = MarkovChainModel(
            n_states=n_states,
            state_names=state_names,
            settings=settings,
            random_seed=random_seed,
        )
        self.meta = RegimeModelMeta(
            name="markov_chain",
            version="1.0.0",
            description=self.meta.description,
            n_states=self._engine.meta.n_states,
            algorithm_family="markov",
            parameters=dict(self._engine.meta.parameters),
            state_names=self._engine.state_names,
        )
        self._state_names = self._engine.state_names

    def fit(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> MarkovRegimeModel:
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
