"""Institutional Bayesian Regime Switching Model (State Space + Regime adapters)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.regimes.base.forecast import RegimeForecast
from iqrp.app.regimes.base.regime_model import RegimeModel, RegimeModelMeta
from iqrp.app.regimes.base.registry import register_regime_model
from iqrp.app.regimes.bayesian.config import BayesianSettings
from iqrp.app.regimes.bayesian.diagnostics import BayesianDiagnostics
from iqrp.app.regimes.bayesian.emissions import BayesianEmissions
from iqrp.app.regimes.bayesian.evaluator import BayesianEvaluator
from iqrp.app.regimes.bayesian.inference import smoothed_state_probabilities
from iqrp.app.regimes.bayesian.posterior import (
    Posterior,
    posterior_predictive_observations,
)
from iqrp.app.regimes.bayesian.prediction import (
    current_state_distribution,
    forecast_from_posterior,
)
from iqrp.app.regimes.bayesian.priors import ModelPriors
from iqrp.app.regimes.bayesian.serializer import BayesianSerializer
from iqrp.app.regimes.bayesian.trainer import BayesianTrainer, model_comparison_scores
from iqrp.app.regimes.bayesian.transitions import BayesianTransitions
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
class BayesianRegimeSwitchingModel(StateSpaceModel):
    """Bayesian HMM / Markov-switching engine with posterior parameter uncertainty."""

    meta = StateSpaceModelMeta(
        name="bayesian_regime",
        version="1.0.0",
        description="Bayesian regime-switching model with MCMC / VI inference",
        n_states=3,
        algorithm_family="bayesian",
        parameters={},
        state_names=("state_0", "state_1", "state_2"),
    )

    def __init__(
        self,
        *,
        n_states: int | None = None,
        n_features: int | None = None,
        state_names: tuple[str, ...] | None = None,
        settings: BayesianSettings | None = None,
        random_seed: int | None = None,
        user_priors: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._bayes_settings = settings or BayesianSettings.default()
        k = int(n_states if n_states is not None else self._bayes_settings.n_states)
        d = int(n_features if n_features is not None else self._bayes_settings.n_features)
        names = state_names if state_names is not None else self._bayes_settings.state_names
        self._state_names = _resolve_names(k, names)
        self._rng = np.random.default_rng(
            random_seed if random_seed is not None else self._bayes_settings.random_seed
        )
        self.priors = ModelPriors.from_config(
            self._bayes_settings.priors, k, d, user_priors=user_priors
        )
        self._posterior: Posterior | None = None
        self.transitions: BayesianTransitions | None = None
        self.emissions: BayesianEmissions | None = None
        self._history: list[float] = []
        self._acceptance_rate: float | None = None
        self._n_iter = 0
        self._train_obs: np.ndarray | None = None
        self._online_buffer: list[np.ndarray] = []
        self._update_counter = 0
        self.meta = StateSpaceModelMeta(
            name="bayesian_regime",
            version="1.0.0",
            description=self.meta.description,
            n_states=k,
            algorithm_family="bayesian",
            parameters={
                "n_states": k,
                "n_features": d,
                "model_type": self._bayes_settings.model_type,
                "algorithm": self._bayes_settings.inference.algorithm,
                "emission": self._bayes_settings.emission.type,
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
    ) -> BayesianRegimeSwitchingModel:
        y = self._extract_obs(observations, observation_columns)
        d = int(y.shape[1]) if y.ndim == 2 else 1
        if self.priors.mean_loc.shape[-1] != d:
            self.priors = ModelPriors.from_config(
                self._bayes_settings.priors, self.meta.n_states, d
            )
            self.meta = StateSpaceModelMeta(
                name=self.meta.name,
                version=self.meta.version,
                description=self.meta.description,
                n_states=self.meta.n_states,
                algorithm_family=self.meta.algorithm_family,
                parameters={**dict(self.meta.parameters), "n_features": d},
                state_names=self._state_names,
            )
        result = BayesianTrainer(self._bayes_settings).fit(
            y, n_states=self.meta.n_states, priors=self.priors, rng=self._rng
        )
        self._ingest_result(result, y)
        return self

    def partial_fit(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> BayesianRegimeSwitchingModel:
        y = self._extract_obs(observations, observation_columns)
        if not self._fitted or self._posterior is None:
            return self.fit(observations, observation_columns=observation_columns)
        s = self._bayes_settings
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
        warm = None
        if self.transitions is not None and self.emissions is not None:
            last_states = (
                self._posterior.draws[-1].states
                if self._posterior.draws and self._posterior.draws[-1].states is not None
                else np.zeros(chunk.shape[0], dtype=np.int64)
            )
            if last_states is not None and last_states.size != chunk.shape[0]:
                last_states = np.resize(last_states, chunk.shape[0])
            warm = (self.transitions, self.emissions, last_states)
        # short warm-start Gibbs refinement
        from iqrp.app.regimes.bayesian.gibbs import run_gibbs

        result = run_gibbs(
            chunk,
            self.meta.n_states,
            self.priors,
            covariance_type=s.emission.covariance_type,
            n_chains=1,
            n_samples=max(20, s.inference.n_samples // 5),
            burn_in=max(5, s.inference.burn_in // 5),
            thin=1,
            n_jobs=1,
            warm_start=warm,
            rng=self._rng,
        )
        self._ingest_result(result, chunk)
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
        gamma, ll = smoothed_state_probabilities(
            log_e, self.transitions.transition, self.transitions.initial
        )
        states = np.argmax(gamma, axis=1).astype(np.int64)
        return FilterResult(
            filtered_states=states,
            filtered_probabilities=gamma,
            log_likelihood=ll,
            normalization_constants=np.ones(y.shape[0]),
        )

    def smooth(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
        lag: int | None = None,
    ) -> SmootherResult:
        filt = self.filter(observations, observation_columns=observation_columns)
        return SmootherResult(
            smoothed_states=filt.filtered_states,
            smoothed_probabilities=filt.filtered_probabilities,
            backward_messages=filt.filtered_probabilities,
            log_likelihood=filt.log_likelihood,
            metadata={"lag": lag},
        )

    def predict(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        return self.smooth(observations, observation_columns=observation_columns).smoothed_states

    def predict_proba(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        assert self._posterior is not None
        y = self._extract_obs(observations, observation_columns)
        # prefer empirical posterior state paths when available and length matches
        proba = self._posterior.posterior_state_probabilities(y.shape[0])
        if (
            proba.shape[0] == y.shape[0]
            and self._posterior.draws
            and any(
                d.states is not None and d.states.size == y.shape[0] for d in self._posterior.draws
            )
        ):
            return proba
        return self.smooth(
            observations, observation_columns=observation_columns
        ).smoothed_probabilities

    def posterior_summary(self) -> Posterior:
        self._require_fitted()
        assert self._posterior is not None
        return self._posterior

    def posterior(self) -> Posterior:
        return self.posterior_summary()

    def posterior_predictive(
        self,
        *,
        n_steps: int = 50,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        self._require_fitted()
        assert self._posterior is not None
        return posterior_predictive_observations(
            self._posterior, n_steps=n_steps, rng=rng or self._rng
        )

    def credible_intervals(
        self,
        parameter: str = "means",
        *,
        level: float | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        assert self._posterior is not None
        lvl = (
            float(level)
            if level is not None
            else float(self._bayes_settings.forecasting.confidence_level)
        )
        return self._posterior.credible_intervals(parameter, level=lvl)

    def sample_posterior(self, n: int | None = None) -> list[dict[str, Any]]:
        self._require_fitted()
        assert self._posterior is not None
        draws = self._posterior.draws
        if n is not None:
            draws = draws[: max(1, int(n))]
        return [d.to_dict() for d in draws]

    def forecast(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
        horizon: int | None = None,
    ) -> ForecastResult:
        self._require_fitted()
        assert self._posterior is not None
        default_h = self._bayes_settings.forecasting.default_horizon
        h = int(horizon if horizon is not None else default_h)
        proba = self.predict_proba(observations, observation_columns=observation_columns)
        current = current_state_distribution(proba)
        return forecast_from_posterior(
            self._posterior,
            current,
            horizon=h,
            state_names=self._state_names,
            confidence_level=self._bayes_settings.forecasting.confidence_level,
            n_draws=self._bayes_settings.forecasting.n_posterior_draws,
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
        k = self.meta.n_states
        p = self.transitions.transition
        pi = self.transitions.initial
        state = int(initial_state if initial_state is not None else rng.choice(k, p=pi))
        states = np.empty(n_steps, dtype=np.int64)
        obs = np.empty((n_steps, self.emissions.n_features), dtype=np.float64)
        for t in range(n_steps):
            states[t] = state
            mu = self.emissions.means[state]
            if self.emissions.covariance_type == "diag":
                obs[t] = rng.normal(mu, np.sqrt(np.clip(self.emissions.covars[state], 1e-12, None)))
            else:
                obs[t] = rng.multivariate_normal(mu, self.emissions.covars[state])
            state = int(rng.choice(k, p=p[state]))
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

    def compare_models(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        y = self._extract_obs(observations, observation_columns)
        return BayesianTrainer(self._bayes_settings).compare_models(y, rng=self._rng)

    def evaluate(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        true_states: np.ndarray | None = None,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        assert self._posterior is not None
        y = self._extract_obs(observations, observation_columns)
        pred = self.predict(observations, observation_columns=observation_columns)
        proba = self.predict_proba(observations, observation_columns=observation_columns)
        return BayesianEvaluator().evaluate(
            true_states=true_states,
            predicted_states=pred,
            probabilities=proba,
            posterior=self._posterior,
            observations=y,
        )

    def diagnostics(
        self,
        observations: pl.DataFrame | np.ndarray | None = None,
        *,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        assert self._posterior is not None
        proba = None
        if observations is not None:
            proba = self.predict_proba(observations, observation_columns=observation_columns)
        elif self._train_obs is not None:
            proba = self.predict_proba(self._train_obs)
        else:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("No observations for diagnostics", code="BAYES_NO_OBS")
        return BayesianDiagnostics().report(
            self._posterior,
            history=self._history,
            acceptance_rate=self._acceptance_rate,
            state_proba=proba,
        )

    def transition_matrix(self) -> np.ndarray:
        self._require_fitted()
        assert self.transitions is not None
        return self.transitions.transition.copy()

    def save(self, path: Path | str) -> Path:
        return BayesianSerializer().save(self, Path(path))

    @classmethod
    def load(cls, path: Path | str) -> BayesianRegimeSwitchingModel:
        return BayesianSerializer().load(Path(path), model_cls=cls)

    def _ingest_result(self, result: Any, y: np.ndarray) -> None:
        self._posterior = result.posterior
        self._history = list(
            getattr(result, "history", None) or getattr(result, "elbo_history", []) or []
        )
        self._acceptance_rate = getattr(result, "acceptance_rate", None)
        self._n_iter = int(getattr(result, "n_iter", 0))
        mean_tm = self._posterior.mean_transition()
        mean_pi = self._posterior.mean_initial()
        mean_mu = self._posterior.mean_means()
        mean_cov = self._posterior.mean_covars()
        cov_type = self._bayes_settings.emission.covariance_type
        self.transitions = BayesianTransitions(
            n_states=self.meta.n_states,
            transition=mean_tm,
            initial=mean_pi,
            prior_alpha=self.priors.transition_alpha,
            prior_initial=self.priors.initial_alpha,
        )
        self.emissions = BayesianEmissions(
            n_states=self.meta.n_states,
            n_features=mean_mu.shape[1] if mean_mu.ndim == 2 else 1,
            means=mean_mu if mean_mu.ndim == 2 else mean_mu.reshape(-1, 1),
            covars=mean_cov,
            covariance_type=cov_type,
            priors=self.priors,
        )
        self._train_obs = y
        self._fitted = True
        scores = model_comparison_scores(y, self._posterior)
        self._posterior.metadata.update(
            {"n_states": self.meta.n_states, "n_features": y.shape[1], **scores}
        )

    def _extract_obs(
        self,
        observations: pl.DataFrame | np.ndarray,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        if isinstance(observations, np.ndarray):
            y = np.asarray(observations, dtype=np.float64)
            return y.reshape(-1, 1) if y.ndim == 1 else y
        cols = observation_columns
        if not cols:
            if self._bayes_settings.columns.observation_columns:
                cols = list(self._bayes_settings.columns.observation_columns)
            else:
                exclude = {
                    self._bayes_settings.columns.timestamp,
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

            raise ValidationError("No observation columns", code="BAYES_NO_OBS_COLS")
        return observations.select(cols).to_numpy().astype(np.float64)

    def _algorithm_state(self) -> dict[str, Any]:
        assert (
            self._posterior is not None
            and self.transitions is not None
            and self.emissions is not None
        )
        return {
            "posterior": self._posterior.to_dict(),
            "transitions": self.transitions.to_dict(),
            "emissions": self.emissions.to_dict(),
            "priors": self.priors.to_dict(),
            "mean_transition": self.transitions.transition.tolist(),
            "mean_initial": self.transitions.initial.tolist(),
            "mean_means": self.emissions.means.tolist(),
            "mean_covars": self.emissions.covars.tolist(),
            "history": list(self._history),
            "acceptance_rate": self._acceptance_rate,
            "n_iter": self._n_iter,
            "state_names": list(self._state_names),
            "settings": self._bayes_settings.model_dump(),
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        if "settings" in state:
            self._bayes_settings = BayesianSettings.from_mapping(state["settings"])
        if "priors" in state:
            self.priors = ModelPriors.from_dict(state["priors"])
        if "posterior" in state:
            self._posterior = Posterior.from_dict(state["posterior"])
        if "transitions" in state:
            self.transitions = BayesianTransitions.from_dict(state["transitions"])
        if "emissions" in state:
            self.emissions = BayesianEmissions.from_dict(state["emissions"])
        self._history = list(state.get("history") or [])
        self._acceptance_rate = state.get("acceptance_rate")
        self._n_iter = int(state.get("n_iter", 0))
        names = state.get("state_names")
        if names:
            self._state_names = tuple(names)
        k = self.transitions.n_states if self.transitions else self.meta.n_states
        self.meta = StateSpaceModelMeta(
            name=self.meta.name,
            version=self.meta.version,
            description=self.meta.description,
            n_states=k,
            algorithm_family="bayesian",
            parameters={
                "n_states": k,
                "algorithm": self._bayes_settings.inference.algorithm,
            },
            state_names=self._state_names,
        )


@register_regime_model
class BayesianRegimeModel(RegimeModel):
    """RegimeModel adapter over :class:`BayesianRegimeSwitchingModel`."""

    meta = RegimeModelMeta(
        name="bayesian_regime",
        version="1.0.0",
        description="Bayesian regime-switching detector",
        n_states=3,
        algorithm_family="bayesian",
        parameters={},
        state_names=("state_0", "state_1", "state_2"),
    )

    def __init__(
        self,
        *,
        n_states: int | None = None,
        settings: BayesianSettings | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self._engine = BayesianRegimeSwitchingModel(
            n_states=n_states, settings=settings, random_seed=random_seed
        )
        self.meta = RegimeModelMeta(
            name="bayesian_regime",
            version="1.0.0",
            description=self.meta.description,
            n_states=self._engine.meta.n_states,
            algorithm_family="bayesian",
            parameters=dict(self._engine.meta.parameters),
            state_names=self._engine.state_names,
        )
        self._state_names = self._engine.state_names

    def fit(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> BayesianRegimeModel:
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
