"""Mock discrete state-space model for framework validation (not a financial model)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.math.stochastic.markov_utils import empirical_transition, simulate_markov
from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.base.forecast_result import ForecastResult
from iqrp.app.state_space.base.observation_model import DiagonalGaussianObservationModel
from iqrp.app.state_space.base.registry import register_state_space_model
from iqrp.app.state_space.base.smoother_result import SmootherResult
from iqrp.app.state_space.base.state_space_model import StateSpaceModel, StateSpaceModelMeta
from iqrp.app.state_space.base.transition_model import MatrixTransitionModel
from iqrp.app.state_space.config import StateSpaceSettings
from iqrp.app.state_space.filtering.forward_filter import ForwardFilter
from iqrp.app.state_space.forecasting.multi_step import MultiStepForecaster
from iqrp.app.state_space.smoothing.fixed_interval import FixedIntervalSmoother
from iqrp.app.state_space.smoothing.fixed_lag import FixedLagSmoother


def _as_array(
    observations: pl.DataFrame | np.ndarray,
    observation_columns: list[str] | None,
    settings: StateSpaceSettings,
) -> np.ndarray:
    if isinstance(observations, np.ndarray):
        y = np.asarray(observations, dtype=np.float64)
        return y.reshape(-1, 1) if y.ndim == 1 else y
    cols = observation_columns
    if not cols:
        if settings.columns.observation_columns:
            cols = list(settings.columns.observation_columns)
        else:
            exclude = {
                settings.columns.timestamp,
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
                "exchange",
                "timeframe",
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

        raise ValidationError("No observation columns available", code="SS_NO_OBSERVATIONS")
    return observations.select(cols).to_numpy().astype(np.float64)


@register_state_space_model
class MockDiscreteStateSpaceModel(StateSpaceModel):
    """Quantile-clustered Gaussian emission SSM for plumbing tests.

    Fits a discrete latent variable model with diagonal Gaussian emissions
    and an empirical transition matrix. Exists so the framework can be
    exercised without committing to HMM / Kalman / particle algorithms.
    """

    meta = StateSpaceModelMeta(
        name="mock_discrete_ssm",
        version="1.0.0",
        description="Mock discrete Gaussian-emission state-space model",
        n_states=3,
        algorithm_family="mock",
        parameters={"n_states": 3, "n_params": 12},
        state_names=("state_0", "state_1", "state_2"),
    )

    def __init__(
        self,
        *,
        n_states: int | None = None,
        random_seed: int | None = None,
        settings: StateSpaceSettings | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings or StateSpaceSettings.default()
        k = int(n_states if n_states is not None else self.meta.n_states)
        seed = random_seed if random_seed is not None else self._settings.random_seed
        self._rng = np.random.default_rng(seed)
        self._params = {"n_states": k, "n_params": k * k + 2 * k}
        self._state_names = tuple(f"state_{i}" for i in range(k))
        self.meta = StateSpaceModelMeta(
            name=self.meta.name,
            version=self.meta.version,
            description=self.meta.description,
            n_states=k,
            algorithm_family=self.meta.algorithm_family,
            parameters=dict(self._params),
            state_names=self._state_names,
        )
        self._transition: MatrixTransitionModel | None = None
        self._observation: DiagonalGaussianObservationModel | None = None
        self._initial: np.ndarray | None = None

    def fit(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> MockDiscreteStateSpaceModel:
        y = _as_array(observations, observation_columns, self._settings)
        k = int(self._params["n_states"])
        primary = y[:, 0]
        finite = primary[np.isfinite(primary)]
        if finite.size < k + 2:
            edges = np.linspace(float(np.nanmin(primary)), float(np.nanmax(primary) + 1e-6), k + 1)
        else:
            edges = np.unique(np.quantile(finite, np.linspace(0, 1, k + 1)))
            if len(edges) < k + 1:
                edges = np.linspace(float(finite.min()), float(finite.max()), k + 1)
        hard = np.clip(np.digitize(primary, edges[1:-1], right=True), 0, k - 1).astype(np.int64)
        hard[~np.isfinite(primary)] = k // 2

        means = np.zeros((k, y.shape[1]), dtype=np.float64)
        vars_ = np.ones((k, y.shape[1]), dtype=np.float64)
        for s in range(k):
            mask = hard == s
            if np.any(mask):
                means[s] = np.nanmean(y[mask], axis=0)
                vars_[s] = np.nanvar(y[mask], axis=0) + 1e-6
            else:
                means[s] = np.nanmean(y, axis=0)
                vars_[s] = np.nanvar(y, axis=0) + 1e-6

        tm = empirical_transition(hard, n_states=k)
        # Dirichlet-style smoothing
        tm = normalize_rows(tm + 1e-3)
        self._transition = MatrixTransitionModel(tm)
        self._observation = DiagonalGaussianObservationModel(means, vars_)
        counts = np.bincount(hard, minlength=k).astype(np.float64)
        self._initial = counts / max(float(counts.sum()), 1.0)
        self._fitted = True
        return self

    def filter(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> FilterResult:
        self._require_fitted()
        assert self._observation is not None and self._transition is not None
        y = _as_array(observations, observation_columns, self._settings)
        log_e = self._observation.log_emission_matrix(y)
        return ForwardFilter(self._settings).run(
            log_e, self._transition.transition_matrix(), initial=self._initial
        )

    def smooth(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
        lag: int | None = None,
    ) -> SmootherResult:
        self._require_fitted()
        assert self._observation is not None and self._transition is not None
        y = _as_array(observations, observation_columns, self._settings)
        log_e = self._observation.log_emission_matrix(y)
        filt = self.filter(observations, observation_columns=observation_columns)
        algo = self._settings.smoothing.algorithm
        if algo == "fixed_lag" or lag is not None:
            return FixedLagSmoother(self._settings).run(
                log_e,
                self._transition.transition_matrix(),
                initial=self._initial,
                filter_result=filt,
                lag=lag,
            )
        return FixedIntervalSmoother(self._settings).run(
            log_e,
            self._transition.transition_matrix(),
            initial=self._initial,
            filter_result=filt,
        )

    def predict(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        return self.filter(observations, observation_columns=observation_columns).filtered_states

    def predict_proba(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        return self.filter(
            observations, observation_columns=observation_columns
        ).filtered_probabilities

    def forecast(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        horizon: int | None = None,
        observation_columns: list[str] | None = None,
    ) -> ForecastResult:
        self._require_fitted()
        assert self._transition is not None
        proba = self.predict_proba(observations, observation_columns=observation_columns)
        return MultiStepForecaster(self._settings).forecast(
            proba[-1],
            self._transition,
            horizon=horizon,
            state_names=self._state_names,
        )

    def sample(
        self,
        n_steps: int,
        *,
        initial_state: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        assert self._transition is not None and self._observation is not None
        rng = rng or self._rng
        states = simulate_markov(
            self._transition.transition_matrix(),
            n_steps,
            initial=initial_state,
            rng=rng,
        )
        obs = np.vstack([self._observation.sample_observation(int(s), rng=rng) for s in states])
        return states, obs

    def log_likelihood(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        return float(
            self.filter(observations, observation_columns=observation_columns).log_likelihood
        )

    def _transition_matrix_or_none(self) -> np.ndarray | None:
        if self._transition is None:
            return None
        return self._transition.transition_matrix()

    def _n_params(self) -> int:
        return int(self._params.get("n_params", self.meta.n_states))

    def _algorithm_state(self) -> dict[str, Any]:
        assert self._transition is not None and self._observation is not None
        return {
            "params": dict(self._params),
            "transition_matrix": self._transition.transition_matrix().tolist(),
            "means": self._observation._means.tolist(),
            "variances": self._observation._variances.tolist(),
            "initial": None if self._initial is None else self._initial.tolist(),
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        self._params = dict(state.get("params") or self._params)
        tm = state.get("transition_matrix")
        means = state.get("means")
        variances = state.get("variances")
        initial = state.get("initial")
        if tm is not None:
            self._transition = MatrixTransitionModel(tm)
        if means is not None and variances is not None:
            self._observation = DiagonalGaussianObservationModel(means, variances)
        self._initial = None if initial is None else np.asarray(initial, dtype=np.float64)
        k = int(self._params.get("n_states", self.meta.n_states))
        self._state_names = tuple(f"state_{i}" for i in range(k))
        self.meta = StateSpaceModelMeta(
            name=self.meta.name,
            version=self.meta.version,
            description=self.meta.description,
            n_states=k,
            algorithm_family=self.meta.algorithm_family,
            parameters=dict(self._params),
            state_names=self._state_names,
        )
