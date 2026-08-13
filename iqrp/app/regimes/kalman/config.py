"""Hydra-backed configuration for the Kalman Filtering Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class SystemConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    dt: float = 1.0
    process_noise_scale: float = 1.0e-3
    observation_noise_scale: float = 1.0e-2
    initial_state_scale: float = 1.0
    initial_covariance_scale: float = 1.0


class UKFConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    alpha: float = 1.0e-3
    beta: float = 2.0
    kappa: float = 0.0


class AdaptiveConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    window: int = 20
    process_adapt_rate: float = 0.05
    observation_adapt_rate: float = 0.05
    innovation_threshold: float = 3.0


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    em_iterations: int = 10
    tol: float = 1.0e-4
    estimate_noise: bool = True


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
    control_columns: tuple[str, ...] | None = None


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_points: int = 500


class SerializationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    format: Literal["json"] = "json"


class KalmanSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    n_states: int = 2
    n_obs: int = 1
    state_names: tuple[str, ...] | None = None
    random_seed: int = 42
    store_dir: str = "data/kalman"
    output_dir: str = "data/reports/kalman"
    filter_type: Literal["linear", "ekf", "ukf", "adaptive"] = "linear"
    application: Literal[
        "custom", "trend", "denoise", "dynamic_beta", "volatility", "spread", "pairs"
    ] = "custom"
    system: SystemConfig = Field(default_factory=SystemConfig)
    ukf: UKFConfig = Field(default_factory=UKFConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | Any) -> KalmanSettings:
        from iqrp.app.core.exceptions import ConfigurationError

        if not isinstance(data, dict):
            if OmegaConf.is_config(data):
                container = OmegaConf.to_container(data, resolve=True)
            else:
                container = data
            if not isinstance(container, dict):
                raise ConfigurationError(
                    "Kalman config mapping invalid", code="KF_CONFIG_INVALID"
                )
            data = container
        return cls.model_validate(data)

    @classmethod
    def from_hydra(
        cls,
        config_path: Path | None = None,
        overrides: list[str] | None = None,
    ) -> KalmanSettings:
        path = config_path or _default_config_path()
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "Kalman config root must be a mapping", code="KF_CONFIG_INVALID"
            )
        return cls.from_mapping(container)

    @classmethod
    def default(cls) -> KalmanSettings:
        path = _default_config_path()
        if path.exists():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "kalman" / "default.yaml"
