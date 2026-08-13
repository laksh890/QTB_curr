"""Hydra-backed configuration for the Bayesian Regime Switching Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class EmissionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["gaussian", "multivariate_gaussian"] = "gaussian"
    covariance_type: Literal["diag", "full"] = "diag"


class PriorsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    transition_alpha: float = 1.0
    initial_alpha: float = 1.0
    mean_prior_strength: float = 0.1
    mean_prior_location: float = 0.0
    invgamma_shape: float = 2.0
    invgamma_scale: float = 1.0
    wishart_df: float = 3.0
    wishart_scale: float = 1.0


class InferenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: Literal["gibbs", "metropolis", "hmc", "variational"] = "gibbs"
    n_chains: int = 2
    n_samples: int = 200
    burn_in: int = 50
    thin: int = 1
    target_accept: float = 0.65
    step_size: float = 0.05
    leapfrog_steps: int = 10
    n_jobs: int = 2
    checkpoint_every: int = 0
    resume: bool = False


class VariationalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_iter: int = 100
    tol: float = 1.0e-4
    learning_rate: float = 0.1


class OnlineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    window_size: int = 0
    update_frequency: int = 1
    warm_start: bool = True


class ForecastingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_horizon: int = 5
    confidence_level: float = 0.95
    n_posterior_draws: int = 50


class ModelComparisonConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_states: int = 2
    max_states: int = 4
    criterion: Literal["waic", "loo", "marginal_likelihood"] = "waic"


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    observation_columns: tuple[str, ...] | None = None


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_points: int = 500


class SerializationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    format: Literal["json"] = "json"


class BayesianSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    n_states: int = 3
    n_features: int = 1
    state_names: tuple[str, ...] | None = None
    random_seed: int = 42
    store_dir: str = "data/bayesian"
    output_dir: str = "data/reports/bayesian"
    model_type: Literal["bayesian_hmm", "bayesian_markov_switching"] = "bayesian_hmm"
    emission: EmissionConfig = Field(default_factory=EmissionConfig)
    priors: PriorsConfig = Field(default_factory=PriorsConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    variational: VariationalConfig = Field(default_factory=VariationalConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    model_comparison: ModelComparisonConfig = Field(default_factory=ModelComparisonConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | Any) -> BayesianSettings:
        from iqrp.app.core.exceptions import ConfigurationError

        if not isinstance(data, dict):
            if OmegaConf.is_config(data):
                container = OmegaConf.to_container(data, resolve=True)
            else:
                container = data
            if not isinstance(container, dict):
                raise ConfigurationError(
                    "Bayesian config mapping invalid", code="BAYES_CONFIG_INVALID"
                )
            data = container
        return cls.model_validate(data)

    @classmethod
    def from_hydra(
        cls,
        config_path: Path | None = None,
        overrides: list[str] | None = None,
    ) -> BayesianSettings:
        path = config_path or _default_config_path()
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "Bayesian config root must be a mapping", code="BAYES_CONFIG_INVALID"
            )
        return cls.from_mapping(container)

    @classmethod
    def default(cls) -> BayesianSettings:
        path = _default_config_path()
        if path.exists():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "bayesian" / "default.yaml"
