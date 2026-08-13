"""Hydra-backed configuration for the State Space Framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    observation_columns: tuple[str, ...] | None = None


class FilteringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: Literal["forward", "backward"] = "forward"
    numerical_eps: float = 1.0e-300
    chunk_size: int = 10_000


class SmoothingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: Literal["fixed_interval", "fixed_lag"] = "fixed_interval"
    fixed_lag: int = 5


class ForecastingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_horizon: int = 5
    confidence_level: float = 0.95


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_bootstrap: int = 0


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_points: int = 500


class StorageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    compression: str = "zstd"
    register_duckdb: bool = True


class SerializationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    format: Literal["json", "parquet_sidecar"] = "json"


class StateSpaceSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    n_jobs: int = 4
    random_seed: int = 42
    store_dir: str = "data/state_space"
    duckdb_path: str = "data/state_space/state_space.duckdb"
    output_dir: str = "data/reports/state_space"
    default_model: str = "mock_discrete_ssm"
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    filtering: FilteringConfig = Field(default_factory=FilteringConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | Any) -> StateSpaceSettings:
        from iqrp.app.core.exceptions import ConfigurationError

        if not isinstance(data, dict):
            if OmegaConf.is_config(data):
                container = OmegaConf.to_container(data, resolve=True)
            else:
                container = data
            if not isinstance(container, dict):
                raise ConfigurationError(
                    "State-space config mapping invalid",
                    code="SS_CONFIG_INVALID",
                )
            data = container
        return cls.model_validate(data)

    @classmethod
    def from_hydra(
        cls,
        config_path: Path | None = None,
        overrides: list[str] | None = None,
    ) -> StateSpaceSettings:
        path = config_path or _default_config_path()
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "State-space config root must be a mapping",
                code="SS_CONFIG_INVALID",
            )
        return cls.from_mapping(container)

    @classmethod
    def default(cls) -> StateSpaceSettings:
        path = _default_config_path()
        if path.exists():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "state_space" / "default.yaml"
