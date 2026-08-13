"""Institutional Gaussian Mixture Regime Detection (State Space + Regime adapters)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.math.probability.likelihood import aic as aic_score, bic as bic_score
from iqrp.app.regimes.base.forecast import RegimeForecast
from iqrp.app.regimes.base.regime_model import RegimeModel, RegimeModelMeta
from iqrp.app.regimes.base.registry import register_regime_model
from iqrp.app.regimes.gmm.config import GMMSettings
from iqrp.app.regimes.gmm.diagnostics import GMMDiagnostics
from iqrp.app.regimes.gmm.em import fit_em
from iqrp.app.regimes.gmm.evaluator import GMMEvaluator
from iqrp.app.regimes.gmm.mixture import (
    GaussianMixtureParams,
    PreprocessState,
    transform_preprocess,
)
from iqrp.app.regimes.gmm.prediction import (
    detect_outliers,
    forecast_occupancy,
    hard_assignments,
    soft_assignments,
)
from iqrp.app.regimes.gmm.serializer import GMMSerializer
from iqrp.app.regimes.gmm.trainer import GMMTrainer
from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.base.forecast_result import ForecastResult
from iqrp.app.state_space.base.registry import register_state_space_model
from iqrp.app.state_space.base.smoother_result import SmootherResult
from iqrp.app.state_space.base.state_space_model import StateSpaceModel, StateSpaceModelMeta


def _resolve_names(n_states: int, names: tuple[str, ...] | None) -> tuple[str, ...]:
    if names and len(names) >= n_states:
        return tuple(names[:n_states])
    if names:
        return tuple(names) + tuple(f"regime_{i}" for i in range(len(names), n_states))
    return tuple(f"regime_{i}" for i in range(n_states))


@register_state_space_model
class GaussianMixtureModel(StateSpaceModel):
    """Production GMM / Bayesian GMM for soft market regime detection."""

    meta = StateSpaceModelMeta(
        name="gmm",
        version="1.0.0",
        description="Gaussian Mixture Model for market regime detection",
        n_states=3,
        algorithm_family="gmm",
        parameters={},
        state_names=("regime_0", "regime_1", "regime_2"),
    )

    def __init__(
        self,
        *,
        n_components: int | None = None,
        n_features: int | None = None,
        state_names: tuple[str, ...] | None = None,
        settings: GMMSettings | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self._gmm_settings = settings or GMMSettings.default()
        k = int(n_components if n_components is not None else self._gmm_settings.n_components)
        d = int(n_features if n_features is not None else self._gmm_settings.n_features)
        names = state_names if state_names is not None else self._gmm_settings.state_names
        self._state_names = _resolve_names(k, names)
        self._rng = np.random.default_rng(
            random_seed if random_seed is not None else self._gmm_settings.random_seed
        )
        self.params: GaussianMixtureParams | None = None
        self._responsibilities: np.ndarray | None = None
        self._history: list[float] = []
        self._n_iter = 0
        self._converged = False
        self._train_obs: np.ndarray | None = None
        self._online_buffer: list[np.ndarray] = []
        self._update_counter = 0
        self.meta = StateSpaceModelMeta(
            name="gmm",
            version="1.0.0",
            description=self.meta.description,
            n_states=k,
            algorithm_family="gmm",
            parameters={
                "n_components": k,
                "n_features": d,
                "model_type": self._gmm_settings.model_type,
                "covariance_type": self._gmm_settings.covariance.type,
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
    ) -> GaussianMixtureModel:
        y = self._extract_obs(observations, observation_columns)
        result, prep = GMMTrainer(self._gmm_settings).fit(
            y, n_components=self.meta.n_states, rng=self._rng
        )
        self._ingest(result, prep, y)
        return self

    def partial_fit(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> GaussianMixtureModel:
        y = self._extract_obs(observations, observation_columns)
        if not self._fitted or self.params is None:
            return self.fit(observations, observation_columns=observation_columns)
        s = self._gmm_settings
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
        z = transform_preprocess(chunk, self.params.preprocess)
        warm = (self.params.weights, self.params.means, self.params.covars)
        result = fit_em(
            z,
            self.meta.n_states,
            model_type=s.model_type,
            covariance_type=s.covariance.type,
            max_iter=max(10, s.training.max_iter // 5),
            tol=s.training.tol,
            early_stopping=True,
            reg_covar=(s.covariance.reg_covar * (1.1 if s.online.adaptive_covariance else 1.0)),
            warm_start=warm,
            bayesian_params=s.bayesian.model_dump(),
            rng=self._rng,
        )
        self._ingest(result, self.params.preprocess, chunk)
        self._online_buffer = []
        return self

    def filter(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> FilterResult:
        self._require_fitted()
        assert self.params is not None
        y = self._extract_obs(observations, observation_columns)
        resp = soft_assignments(self.params.responsibilities(y))
        states = hard_assignments(resp)
        ll = float(np.sum(self.params.score_samples(y)))
        return FilterResult(
            filtered_states=states,
            filtered_probabilities=resp,
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
        return self.smooth(
            observations, observation_columns=observation_columns
        ).smoothed_probabilities

    def score(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        return self.log_likelihood(observations, observation_columns=observation_columns)

    def forecast(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
        horizon: int | None = None,
    ) -> ForecastResult:
        self._require_fitted()
        proba = self.predict_proba(observations, observation_columns=observation_columns)
        h = int(horizon if horizon is not None else self._gmm_settings.forecasting.default_horizon)
        return forecast_occupancy(
            proba,
            horizon=h,
            state_names=self._state_names,
            confidence_level=self._gmm_settings.forecasting.confidence_level,
        )

    def sample(
        self,
        n_steps: int,
        *,
        initial_state: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        assert self.params is not None
        comps, obs = self.params.sample(n_steps, rng=rng or self._rng)
        if initial_state is not None and comps.size:
            comps[0] = int(initial_state)
        return comps, obs

    def log_likelihood(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        return float(
            self.filter(observations, observation_columns=observation_columns).log_likelihood
        )

    def aic(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        self._require_fitted()
        assert self.params is not None
        ll = self.log_likelihood(observations, observation_columns=observation_columns)
        return aic_score(-ll, self.params.n_params())

    def bic(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        self._require_fitted()
        assert self.params is not None
        y = self._extract_obs(observations, observation_columns)
        ll = self.log_likelihood(observations, observation_columns=observation_columns)
        return bic_score(-ll, self.params.n_params(), y.shape[0])

    def component_means(self) -> np.ndarray:
        self._require_fitted()
        assert self.params is not None
        return self.params.means.copy()

    def component_covariances(self) -> np.ndarray:
        self._require_fitted()
        assert self.params is not None
        return self.params.expanded_covariances()

    def cluster_statistics(self) -> dict[str, Any]:
        self._require_fitted()
        assert self.params is not None and self._responsibilities is not None
        occ = self._responsibilities.mean(axis=0)
        return {
            "weights": self.params.weights,
            "means": self.params.means,
            "occupancy": occ,
            "n_components": self.params.n_components,
            "n_features": self.params.n_features,
            "covariance_type": self.params.covariance_type,
        }

    def select_model(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        y = self._extract_obs(observations, observation_columns)
        return GMMTrainer(self._gmm_settings).select(y, rng=self._rng)

    def evaluate(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        true_states: np.ndarray | None = None,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        assert self.params is not None
        y = self._extract_obs(observations, observation_columns)
        resp = self.predict_proba(observations, observation_columns=observation_columns)
        ll = self.log_likelihood(observations, observation_columns=observation_columns)
        return GMMEvaluator().evaluate(
            x=y,
            params=self.params,
            responsibilities=resp,
            log_likelihood=ll,
            true_labels=true_states,
        )

    def diagnostics(
        self,
        observations: pl.DataFrame | np.ndarray | None = None,
        *,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        assert self.params is not None
        if observations is not None:
            y = self._extract_obs(observations, observation_columns)
            resp = self.predict_proba(observations, observation_columns=observation_columns)
        elif self._train_obs is not None and self._responsibilities is not None:
            y = self._train_obs
            resp = self._responsibilities
        else:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("No observations for diagnostics", code="GMM_NO_OBS")
        return GMMDiagnostics().report(
            self.params,
            x=y,
            responsibilities=resp,
            history=self._history,
            density_quantile=self._gmm_settings.outlier.density_quantile,
            rare_occupancy=self._gmm_settings.outlier.rare_occupancy,
        )

    def outliers(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        assert self.params is not None
        y = self._extract_obs(observations, observation_columns)
        return detect_outliers(
            self.params, y, density_quantile=self._gmm_settings.outlier.density_quantile
        )

    def transition_matrix(self) -> np.ndarray:
        self._require_fitted()
        assert self._responsibilities is not None
        from iqrp.app.regimes.gmm.prediction import transition_frequency

        hard = hard_assignments(self._responsibilities)
        freq = transition_frequency(hard, self.meta.n_states)
        # row-normalize to stochastic
        row = np.clip(freq.sum(axis=1, keepdims=True), 1e-12, None)
        return np.asarray(freq / row, dtype=np.float64)

    def save(self, path: Path | str) -> Path:
        return GMMSerializer().save(self, Path(path))

    @classmethod
    def load(cls, path: Path | str) -> GaussianMixtureModel:
        return GMMSerializer().load(Path(path), model_cls=cls)

    def _ingest(self, result: Any, prep: PreprocessState, y: np.ndarray) -> None:
        self.params = GaussianMixtureParams(
            weights=result.weights,
            means=result.means,
            covars=result.covars,
            covariance_type=result.covariance_type,
            preprocess=prep,
            model_type=result.model_type,
        )
        self._responsibilities = result.responsibilities
        self._history = list(result.history)
        self._n_iter = int(result.n_iter)
        self._converged = bool(result.converged)
        self._train_obs = y
        self._fitted = True
        k = int(result.weights.shape[0])
        self._state_names = _resolve_names(
            k, self._state_names if len(self._state_names) == k else None
        )
        self.meta = StateSpaceModelMeta(
            name="gmm",
            version="1.0.0",
            description=self.meta.description,
            n_states=k,
            algorithm_family="gmm",
            parameters={
                "n_components": k,
                "n_features": int(result.means.shape[1]),
                "model_type": result.model_type,
                "covariance_type": result.covariance_type,
            },
            state_names=self._state_names,
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
            if self._gmm_settings.columns.observation_columns:
                cols = list(self._gmm_settings.columns.observation_columns)
            else:
                exclude = {
                    self._gmm_settings.columns.timestamp,
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

            raise ValidationError("No observation columns", code="GMM_NO_OBS_COLS")
        return observations.select(cols).to_numpy().astype(np.float64)

    def _algorithm_state(self) -> dict[str, Any]:
        assert self.params is not None
        return {
            "params": self.params.to_dict(),
            "weights": self.params.weights.tolist(),
            "means": self.params.means.tolist(),
            "covars": self.params.covars.tolist(),
            "history": list(self._history),
            "n_iter": self._n_iter,
            "converged": self._converged,
            "state_names": list(self._state_names),
            "settings": self._gmm_settings.model_dump(),
            "responsibilities": (
                None if self._responsibilities is None else self._responsibilities.tolist()
            ),
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        if "settings" in state:
            self._gmm_settings = GMMSettings.from_mapping(state["settings"])
        if "params" in state:
            self.params = GaussianMixtureParams.from_dict(state["params"])
        elif "weights" in state:
            self.params = GaussianMixtureParams(
                weights=np.asarray(state["weights"], dtype=np.float64),
                means=np.asarray(state["means"], dtype=np.float64),
                covars=np.asarray(state["covars"], dtype=np.float64),
                covariance_type=self._gmm_settings.covariance.type,
                preprocess=PreprocessState(),
                model_type=self._gmm_settings.model_type,
            )
        self._history = list(state.get("history") or [])
        self._n_iter = int(state.get("n_iter", 0))
        self._converged = bool(state.get("converged", False))
        resp = state.get("responsibilities")
        self._responsibilities = None if resp is None else np.asarray(resp, dtype=np.float64)
        names = state.get("state_names")
        if names:
            self._state_names = tuple(names)
        k = self.params.n_components if self.params else self.meta.n_states
        self.meta = StateSpaceModelMeta(
            name=self.meta.name,
            version=self.meta.version,
            description=self.meta.description,
            n_states=k,
            algorithm_family="gmm",
            parameters={"n_components": k},
            state_names=self._state_names,
        )


@register_regime_model
class GMMRegimeModel(RegimeModel):
    """RegimeModel adapter over :class:`GaussianMixtureModel`."""

    meta = RegimeModelMeta(
        name="gmm",
        version="1.0.0",
        description="Gaussian Mixture regime detector",
        n_states=3,
        algorithm_family="gmm",
        parameters={},
        state_names=("regime_0", "regime_1", "regime_2"),
    )

    def __init__(
        self,
        *,
        n_components: int | None = None,
        settings: GMMSettings | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self._engine = GaussianMixtureModel(
            n_components=n_components, settings=settings, random_seed=random_seed
        )
        self.meta = RegimeModelMeta(
            name="gmm",
            version="1.0.0",
            description=self.meta.description,
            n_states=self._engine.meta.n_states,
            algorithm_family="gmm",
            parameters=dict(self._engine.meta.parameters),
            state_names=self._engine.state_names,
        )
        self._state_names = self._engine.state_names

    def fit(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> GMMRegimeModel:
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
