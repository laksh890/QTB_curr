"""Institutional Particle Filter Engine (State Space + Regime adapters)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import special  # type: ignore[import-untyped]

from iqrp.app.math.probability.likelihood import aic as aic_score, bic as bic_score
from iqrp.app.regimes.base.forecast import RegimeForecast
from iqrp.app.regimes.base.regime_model import RegimeModel, RegimeModelMeta
from iqrp.app.regimes.base.registry import register_regime_model
from iqrp.app.regimes.particle.config import ParticleSettings
from iqrp.app.regimes.particle.diagnostics import ParticleDiagnostics
from iqrp.app.regimes.particle.evaluator import ParticleEvaluator
from iqrp.app.regimes.particle.particle import FilterTrace, ParticleCloud
from iqrp.app.regimes.particle.prediction import (
    credible_interval,
    forecast_particles,
    posterior_summary,
)
from iqrp.app.regimes.particle.propagation import TransitionModel, build_transition
from iqrp.app.regimes.particle.resampling import apply_resampling
from iqrp.app.regimes.particle.serializer import ParticleSerializer
from iqrp.app.regimes.particle.smoothing import SmoothTrace, trajectory_smooth
from iqrp.app.regimes.particle.trainer import ParticleTrainer, run_filter, simulate_nonlinear
from iqrp.app.regimes.particle.weighting import effective_sample_size
from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.base.forecast_result import ForecastResult
from iqrp.app.state_space.base.registry import register_state_space_model
from iqrp.app.state_space.base.smoother_result import SmootherResult
from iqrp.app.state_space.base.state_space_model import StateSpaceModel, StateSpaceModelMeta

_SOFT_STATE_NAMES = ("bearish", "bullish")


def _soft_trend_proba(means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    m = np.asarray(means, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    primary = m[:, 0]
    std = np.sqrt(np.clip(np.array([c[0, 0] for c in covs], dtype=np.float64), 1e-12, None))
    z = primary / std
    p_bull = 0.5 * (1.0 + special.erf(z / np.sqrt(2.0)))
    p_bull = np.clip(p_bull, 1e-6, 1.0 - 1e-6)
    return np.column_stack([1.0 - p_bull, p_bull])


def _resolve_names(names: tuple[str, ...] | None) -> tuple[str, ...]:
    if names and len(names) >= 2:
        return tuple(names[:2])
    if names:
        return tuple(names) + tuple(_SOFT_STATE_NAMES[len(names) :])
    return _SOFT_STATE_NAMES


@register_state_space_model
class ParticleFilterModel(StateSpaceModel):
    """Production Sequential Monte Carlo engine for nonlinear latent states."""

    meta = StateSpaceModelMeta(
        name="particle",
        version="1.0.0",
        description="Institutional Particle Filter Engine",
        n_states=2,
        algorithm_family="particle",
        parameters={},
        state_names=_SOFT_STATE_NAMES,
    )

    def __init__(
        self,
        *,
        n_states: int | None = None,
        n_obs: int | None = None,
        n_particles: int | None = None,
        state_names: tuple[str, ...] | None = None,
        settings: ParticleSettings | None = None,
        random_seed: int | None = None,
        transition: TransitionModel | None = None,
    ) -> None:
        super().__init__()
        self._pf_settings = settings or ParticleSettings.default()
        self._n_latent = int(n_states if n_states is not None else self._pf_settings.n_states)
        self._n_obs = int(n_obs if n_obs is not None else self._pf_settings.n_obs)
        self._n_particles = int(
            n_particles if n_particles is not None else self._pf_settings.n_particles
        )
        names = state_names if state_names is not None else self._pf_settings.state_names
        self._state_names = _resolve_names(names)
        self._rng = np.random.default_rng(
            random_seed if random_seed is not None else self._pf_settings.random_seed
        )
        self.transition: TransitionModel | None = transition
        self._trace: FilterTrace | None = None
        self._smooth: SmoothTrace | None = None
        self._cloud: ParticleCloud | None = None
        self._history: list[float] = []
        self._train_obs: np.ndarray | None = None
        self.meta = StateSpaceModelMeta(
            name="particle",
            version="1.0.0",
            description=self.meta.description,
            n_states=2,
            algorithm_family="particle",
            parameters={
                "n_latent": self._n_latent,
                "n_obs": self._n_obs,
                "n_particles": self._n_particles,
                "filter_type": self._pf_settings.filter_type,
                "application": self._pf_settings.application,
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
    ) -> ParticleFilterModel:
        y = self._extract_obs(observations, observation_columns)
        settings = self._pf_settings
        if self._n_particles != settings.n_particles:
            settings = ParticleSettings.from_mapping(
                {**settings.model_dump(), "n_particles": self._n_particles}
            )
            self._pf_settings = settings
        model = self.transition or build_transition(
            settings, n_states=self._n_latent, application=settings.application  # type: ignore[arg-type]
        )
        result = ParticleTrainer(settings).fit(y, model=model, rng=self._rng)
        self._ingest(result.model, result.trace, y, result.history)
        return self

    def partial_fit(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> ParticleFilterModel:
        y = self._extract_obs(observations, observation_columns)
        if not self._fitted or self.transition is None:
            return self.fit(observations, observation_columns=observation_columns)
        if not self._pf_settings.online.warm_start:
            return self.fit(observations, observation_columns=observation_columns)
        for t in range(y.shape[0]):
            self.update(y[t])
        if self._train_obs is None:
            self._train_obs = y
        else:
            self._train_obs = np.vstack([self._train_obs, y])
        return self

    def update(self, observation: np.ndarray | float) -> np.ndarray:
        """Single-step online particle update; returns posterior mean."""
        self._require_fitted()
        assert self.transition is not None
        z = np.asarray(observation, dtype=np.float64).reshape(1, -1)
        cloud0 = self._cloud
        trace = run_filter(z, self.transition, self._pf_settings, rng=self._rng, cloud0=cloud0)
        self._cloud = trace.clouds[-1] if trace.clouds else cloud0
        if self._trace is None:
            self._trace = trace
        else:
            self._trace = FilterTrace(
                means=np.vstack([self._trace.means, trace.means]),
                covs=np.concatenate([self._trace.covs, trace.covs], axis=0),
                clouds=list(self._trace.clouds) + list(trace.clouds),
                ess=np.concatenate([self._trace.ess, trace.ess]),
                resampled=np.concatenate([self._trace.resampled, trace.resampled]),
                log_likelihood=self._trace.log_likelihood + trace.log_likelihood,
                metadata=dict(self._trace.metadata),
            )
        return self._cloud.mean() if self._cloud is not None else trace.means[-1]

    def filter(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> FilterResult:
        self._require_fitted()
        assert self.transition is not None
        y = self._extract_obs(observations, observation_columns)
        trace = run_filter(y, self.transition, self._pf_settings, rng=self._rng)
        self._trace = trace
        self._cloud = trace.clouds[-1] if trace.clouds else None
        proba = _soft_trend_proba(trace.means, trace.covs)
        states = np.argmax(proba, axis=1).astype(np.int64)
        return FilterResult(
            filtered_states=states,
            filtered_probabilities=proba,
            log_likelihood=trace.log_likelihood,
            normalization_constants=np.clip(trace.ess, 1e-300, None),
            metadata={
                "means": trace.means,
                "covs": trace.covs,
                "ess": trace.ess,
                "resampled": trace.resampled,
                "filter": self._pf_settings.filter_type,
            },
        )

    def smooth(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
        lag: int | None = None,
    ) -> SmootherResult:
        self._require_fitted()
        assert self.transition is not None
        filt = self.filter(observations, observation_columns=observation_columns)
        assert self._trace is not None
        n_traj = min(self._n_particles, 100)
        sm = trajectory_smooth(self._trace, self.transition, n_trajectories=n_traj, rng=self._rng)
        self._smooth = sm
        proba = _soft_trend_proba(sm.means, sm.covs)
        states = np.argmax(proba, axis=1).astype(np.int64)
        return SmootherResult(
            smoothed_states=states,
            smoothed_probabilities=proba,
            backward_messages=proba,
            log_likelihood=filt.log_likelihood,
            metadata={"lag": lag, "means": sm.means, "covs": sm.covs},
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
        observation_columns: list[str] | None = None,
        horizon: int | None = None,
    ) -> ForecastResult:
        self._require_fitted()
        assert self.transition is not None
        self.filter(observations, observation_columns=observation_columns)
        assert self._cloud is not None
        h = int(horizon if horizon is not None else self._pf_settings.forecasting.default_horizon)
        means, covs, _clouds = forecast_particles(
            self._cloud, self.transition, horizon=h, rng=self._rng
        )
        step_proba = _soft_trend_proba(means, covs)
        terminal = step_proba[-1]
        level = self._pf_settings.forecasting.confidence_level
        lo, hi = credible_interval(self._cloud, level=level, dim=0)
        return ForecastResult(
            horizon=h,
            expected_state=int(np.argmax(terminal)),
            probability_distribution=terminal,
            confidence_interval=(lo, hi),
            expected_duration={0: float(h) * float(terminal[0]), 1: float(h) * float(terminal[1])},
            step_distributions=step_proba,
            state_names=self._state_names,
            metadata={"state_means": means, "state_covs": covs},
        )

    def sample(
        self,
        n_steps: int,
        *,
        initial_state: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        assert self.transition is not None
        states, obs = simulate_nonlinear(
            self.transition,
            n_steps,
            rng=rng or self._rng,
            obs_scale=self._pf_settings.system.observation_noise_scale,
        )
        proba = _soft_trend_proba(states, np.stack([np.eye(states.shape[1])] * n_steps))
        labels = np.argmax(proba, axis=1).astype(np.int64)
        if initial_state is not None and labels.size:
            labels[0] = int(initial_state)
        return labels, obs

    def log_likelihood(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        return float(
            self.filter(observations, observation_columns=observation_columns).log_likelihood
        )

    def score(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        return self.log_likelihood(observations, observation_columns=observation_columns)

    def aic(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        return aic_score(
            -self.log_likelihood(observations, observation_columns=observation_columns),
            self._n_params(),
        )

    def bic(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        y = self._extract_obs(observations, observation_columns)
        return bic_score(
            -self.log_likelihood(observations, observation_columns=observation_columns),
            self._n_params(),
            y.shape[0],
        )

    # --- Particle-specific API ---

    def posterior(self) -> dict[str, Any]:
        self._require_fitted()
        assert self._cloud is not None
        return posterior_summary(self._cloud, level=self._pf_settings.forecasting.confidence_level)

    def credible_interval(self, *, level: float | None = None, dim: int = 0) -> tuple[float, float]:
        self._require_fitted()
        assert self._cloud is not None
        lv = float(level if level is not None else self._pf_settings.forecasting.confidence_level)
        return credible_interval(self._cloud, level=lv, dim=dim)

    def effective_sample_size(self) -> float:
        self._require_fitted()
        if self._cloud is not None:
            return effective_sample_size(self._cloud.weights)
        assert self._trace is not None and self._trace.ess.size
        return float(self._trace.ess[-1])

    def resample(self, method: str | None = None) -> ParticleCloud:
        self._require_fitted()
        assert self._cloud is not None
        m = method or self._pf_settings.resampling.method
        self._cloud = apply_resampling(self._cloud, method=m, rng=self._rng)  # type: ignore[arg-type]
        return self._cloud

    def filtered_means(self) -> np.ndarray:
        self._require_fitted()
        assert self._trace is not None
        return self._trace.means.copy()

    def evaluate(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        true_states: np.ndarray | None = None,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        assert self.transition is not None
        y = self._extract_obs(observations, observation_columns)
        self.filter(observations, observation_columns=observation_columns)
        assert self._trace is not None
        return ParticleEvaluator().evaluate(
            observations=y,
            trace=self._trace,
            true_states=true_states,
            n_params=self._n_params(),
        )

    def diagnostics(
        self,
        observations: pl.DataFrame | np.ndarray | None = None,
        *,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        assert self.transition is not None
        if observations is not None:
            self.filter(observations, observation_columns=observation_columns)
        elif self._trace is None and self._train_obs is not None:
            self.filter(self._train_obs)
        if self._trace is None:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("No observations for diagnostics", code="PF_NO_OBS")
        return ParticleDiagnostics().report(self.transition, self._trace, history=self._history)

    def save(self, path: Path | str) -> Path:
        return ParticleSerializer().save(self, Path(path))

    @classmethod
    def load(cls, path: Path | str) -> ParticleFilterModel:
        return ParticleSerializer().load(Path(path), model_cls=cls)

    def _ingest(
        self,
        model: TransitionModel,
        trace: FilterTrace,
        y: np.ndarray,
        history: list[float],
    ) -> None:
        self.transition = model
        self._trace = trace
        self._cloud = trace.clouds[-1] if trace.clouds else None
        self._smooth = None
        self._history = list(history)
        self._train_obs = y
        self._n_latent = model.n_states
        self._fitted = True
        self.meta = StateSpaceModelMeta(
            name="particle",
            version="1.0.0",
            description=self.meta.description,
            n_states=2,
            algorithm_family="particle",
            parameters={
                "n_latent": self._n_latent,
                "n_obs": self._n_obs,
                "n_particles": self._n_particles,
                "filter_type": self._pf_settings.filter_type,
                "application": model.application,
                "n_params": self._n_params(),
            },
            state_names=self._state_names,
        )

    def _n_params(self) -> int:
        d = self._n_latent
        return int(d * d + 2)

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
            if self._pf_settings.columns.observation_columns:
                cols = list(self._pf_settings.columns.observation_columns)
            else:
                exclude = {
                    self._pf_settings.columns.timestamp,
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

            raise ValidationError("No observation columns", code="PF_NO_OBS_COLS")
        return observations.select(cols).to_numpy().astype(np.float64)

    def _algorithm_state(self) -> dict[str, Any]:
        assert self.transition is not None
        algo: dict[str, Any] = {
            "transition": self.transition.to_dict(),
            "history": list(self._history),
            "state_names": list(self._state_names),
            "settings": self._pf_settings.model_dump(),
            "n_latent": self._n_latent,
            "n_obs": self._n_obs,
            "n_particles": self._n_particles,
        }
        if self._trace is not None:
            algo.update(
                {
                    "means": self._trace.means.tolist(),
                    "covs": self._trace.covs.tolist(),
                    "ess": self._trace.ess.tolist(),
                    "resampled": self._trace.resampled.astype(np.float64).tolist(),
                    "log_likelihood": self._trace.log_likelihood,
                    "trace_metadata": dict(self._trace.metadata),
                }
            )
        if self._cloud is not None:
            algo["cloud_states"] = self._cloud.states.tolist()
            algo["cloud_log_weights"] = self._cloud.log_weights.tolist()
            algo["cloud_likelihoods"] = self._cloud.likelihoods.tolist()
        if self._smooth is not None:
            algo["smooth_means"] = self._smooth.means.tolist()
            algo["smooth_covs"] = self._smooth.covs.tolist()
        return algo

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        if "settings" in state:
            self._pf_settings = ParticleSettings.from_mapping(state["settings"])
        if "transition" in state:
            self.transition = TransitionModel.from_dict(state["transition"])
            # restore application hooks
            rebuilt = build_transition(
                self._pf_settings,
                application=self.transition.application,  # type: ignore[arg-type]
                n_states=self.transition.n_states,
            )
            self.transition = TransitionModel(
                f=self.transition.f,
                q_scale=self.transition.q_scale,
                dt=self.transition.dt,
                application=self.transition.application,
                transition_fn=rebuilt.transition_fn,
                observe_fn=rebuilt.observe_fn,
                metadata=dict(self.transition.metadata),
            )
        self._history = list(state.get("history") or [])
        self._n_latent = int(state.get("n_latent", self._n_latent))
        self._n_obs = int(state.get("n_obs", self._n_obs))
        self._n_particles = int(state.get("n_particles", self._n_particles))
        names = state.get("state_names")
        if names:
            self._state_names = tuple(names)
        if "means" in state:
            # rebuild minimal clouds from means for API continuity
            means = np.asarray(state["means"], dtype=np.float64)
            covs = np.asarray(state["covs"], dtype=np.float64)
            ess = np.asarray(state.get("ess") or [], dtype=np.float64)
            resampled = np.asarray(state.get("resampled") or [], dtype=bool)
            clouds: list[ParticleCloud] = []
            if "cloud_states" in state:
                self._cloud = ParticleCloud(
                    states=np.asarray(state["cloud_states"], dtype=np.float64),
                    log_weights=np.asarray(state["cloud_log_weights"], dtype=np.float64),
                    likelihoods=np.asarray(
                        state.get("cloud_likelihoods") or np.ones(len(state["cloud_states"])),
                        dtype=np.float64,
                    ),
                )
                clouds = [self._cloud]
            self._trace = FilterTrace(
                means=means,
                covs=covs,
                clouds=clouds,
                ess=ess if ess.size else np.zeros(means.shape[0]),
                resampled=resampled if resampled.size else np.zeros(means.shape[0], dtype=bool),
                log_likelihood=float(state.get("log_likelihood", 0.0)),
                metadata=dict(state.get("trace_metadata") or {}),
            )
        if "smooth_means" in state:
            self._smooth = SmoothTrace(
                means=np.asarray(state["smooth_means"], dtype=np.float64),
                covs=np.asarray(state["smooth_covs"], dtype=np.float64),
                trajectories=np.zeros((0, 0, 1)),
            )
        self.meta = StateSpaceModelMeta(
            name=self.meta.name,
            version=self.meta.version,
            description=self.meta.description,
            n_states=2,
            algorithm_family="particle",
            parameters={
                "n_latent": self._n_latent,
                "n_particles": self._n_particles,
                "filter_type": self._pf_settings.filter_type,
            },
            state_names=self._state_names,
        )


@register_regime_model
class ParticleRegimeModel(RegimeModel):
    """RegimeModel adapter over soft trend particle posteriors."""

    meta = RegimeModelMeta(
        name="particle",
        version="1.0.0",
        description="Particle filter soft trend / regime adapter",
        n_states=2,
        algorithm_family="particle",
        parameters={},
        state_names=_SOFT_STATE_NAMES,
    )

    def __init__(
        self,
        *,
        n_states: int | None = None,
        settings: ParticleSettings | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self._engine = ParticleFilterModel(
            n_states=n_states, settings=settings, random_seed=random_seed
        )
        self.meta = RegimeModelMeta(
            name="particle",
            version="1.0.0",
            description=self.meta.description,
            n_states=2,
            algorithm_family="particle",
            parameters=dict(self._engine.meta.parameters),
            state_names=self._engine.state_names,
        )
        self._state_names = self._engine.state_names

    def fit(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> ParticleRegimeModel:
        self._engine.fit(frame, observation_columns=feature_columns)
        self._fitted = True
        self._transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
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
            self._transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
            self._state_names = self._engine.state_names
