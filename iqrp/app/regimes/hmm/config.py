"""Hydra-backed configuration for the Hidden Markov Model Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class EmissionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["discrete", "gaussian", "multivariate_gaussian"] = "gaussian"
    covariance_type: Literal["diag", "full"] = "diag"


class InitializationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["random", "uniform", "kmeans", "user"] = "kmeans"
    n_restarts: int = 3
    dirichlet_alpha: float = 1.0


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_iter: int = 100
    tol: float = 1.0e-4
    early_stopping: bool = True
    min_covar: float = 1.0e-6
    n_jobs: int = 1


class OnlineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    window_size: int = 0
    update_frequency: int = 1
    warm_start: bool = True


class ForecastingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_horizon: int = 5
    confidence_level: float = 0.95


class ModelSelectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_states: int = 2
    max_states: int = 5
    criterion: Literal["aic", "bic", "log_likelihood"] = "bic"


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


class HMMSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    n_states: int = 3
    n_features: int = 1
    state_names: tuple[str, ...] | None = None
    random_seed: int = 42
    store_dir: str = "data/hmm"
    output_dir: str = "data/reports/hmm"
    emission: EmissionConfig = Field(default_factory=EmissionConfig)
    initialization: InitializationConfig = Field(default_factory=InitializationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    model_selection: ModelSelectionConfig = Field(default_factory=ModelSelectionConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | Any) -> HMMSettings:
        from iqrp.app.core.exceptions import ConfigurationError

        if not isinstance(data, dict):
            if OmegaConf.is_config(data):
                container = OmegaConf.to_container(data, resolve=True)
            else:
                container = data
            if not isinstance(container, dict):
                raise ConfigurationError("HMM config mapping invalid", code="HMM_CONFIG_INVALID")
            data = container
        return cls.model_validate(data)

    @classmethod
    def from_hydra(
        cls,
        config_path: Path | None = None,
        overrides: list[str] | None = None,
    ) -> HMMSettings:
        path = config_path or _default_config_path()
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError("HMM config root must be a mapping", code="HMM_CONFIG_INVALID")
        return cls.from_mapping(container)

    @classmethod
    def default(cls) -> HMMSettings:
        path = _default_config_path()
        if path.exists():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "hmm" / "default.yaml"
