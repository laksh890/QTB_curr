"""Hydra-backed configuration for the Particle Filter Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class SystemConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    dt: float = 1.0
    process_noise_scale: float = 1.0e-2
    observation_noise_scale: float = 1.0e-1
    initial_state_scale: float = 1.0
    initial_covariance_scale: float = 1.0
    student_t_df: float = 5.0


class ResamplingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    adaptive: bool = True
    ess_threshold: float = 0.5
    method: Literal["multinomial", "systematic", "residual", "stratified"] = "systematic"


class RejuvenationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    method: Literal["jitter", "mcmc", "adaptive", "covariance"] = "jitter"
    scale: float = 0.05
    mcmc_steps: int = 1


class AdaptiveConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    min_particles: int = 100
    max_particles: int = 5000
    target_ess_fraction: float = 0.5
    proposal_adapt: bool = True


class RaoBlackwellizedConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_linear: int = 1
    kalman_process_noise: float = 1.0e-3
    kalman_observation_noise: float = 1.0e-2


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_iterations: int = 1
    tol: float = 1.0e-4


class OnlineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    warm_start: bool = True
    checkpoint_every: int = 0


class ForecastingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_horizon: int = 5
    confidence_level: float = 0.95


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    observation_columns: tuple[str, ...] | None = None


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_points: int = 500
    max_particles_plot: int = 200


class SerializationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    format: Literal["json"] = "json"


class ParticleSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    n_states: int = 2
    n_obs: int = 1
    n_particles: int = 500
    state_names: tuple[str, ...] | None = None
    random_seed: int = 42
    store_dir: str = "data/particle"
    output_dir: str = "data/reports/particle"
    filter_type: Literal[
        "bootstrap", "sis", "sir", "auxiliary", "rao_blackwellized", "adaptive"
    ] = "bootstrap"
    application: Literal[
        "custom",
        "nonlinear_trend",
        "volatility",
        "liquidity",
        "dynamic_corr",
        "market_stress",
        "risk_factors",
    ] = "custom"
    resampling_method: Literal["multinomial", "systematic", "residual", "stratified"] = "systematic"
    likelihood: Literal["gaussian", "student_t", "laplace", "custom"] = "gaussian"
    system: SystemConfig = Field(default_factory=SystemConfig)
    resampling: ResamplingConfig = Field(default_factory=ResamplingConfig)
    rejuvenation: RejuvenationConfig = Field(default_factory=RejuvenationConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    rao_blackwellized: RaoBlackwellizedConfig = Field(default_factory=RaoBlackwellizedConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | Any) -> ParticleSettings:
        from iqrp.app.core.exceptions import ConfigurationError

        if not isinstance(data, dict):
            if OmegaConf.is_config(data):
                container = OmegaConf.to_container(data, resolve=True)
            else:
                container = data
            if not isinstance(container, dict):
                raise ConfigurationError(
                    "Particle config mapping invalid", code="PF_CONFIG_INVALID"
                )
            data = container
        return cls.model_validate(data)

    @classmethod
    def from_hydra(
        cls,
        config_path: Path | None = None,
        overrides: list[str] | None = None,
    ) -> ParticleSettings:
        path = config_path or _default_config_path()
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "Particle config root must be a mapping", code="PF_CONFIG_INVALID"
            )
        return cls.from_mapping(container)

    @classmethod
    def default(cls) -> ParticleSettings:
        path = _default_config_path()
        if path.exists():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "particle" / "default.yaml"
