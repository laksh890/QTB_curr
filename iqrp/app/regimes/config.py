"""Hydra-backed configuration for the Regime Detection Framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    feature_columns: tuple[str, ...] | None = None


class DetectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_persistence_bars: int = 3
    confidence_threshold: float = 0.55
    smooth_window: int = 1


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


class RegimeSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    n_jobs: int = 4
    random_seed: int = 42
    store_dir: str = "data/regimes"
    duckdb_path: str = "data/regimes/regimes.duckdb"
    output_dir: str = "data/reports/regimes"
    default_model: str = "mock_regime"
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | Any) -> RegimeSettings:
        from iqrp.app.core.exceptions import ConfigurationError

        if not isinstance(data, dict):
            if OmegaConf.is_config(data):
                container = OmegaConf.to_container(data, resolve=True)
            else:
                container = data
            if not isinstance(container, dict):
                raise ConfigurationError(
                    "Regime config mapping invalid",
                    code="REGIME_CONFIG_INVALID",
                )
            data = container
        return cls.model_validate(data)

    @classmethod
    def from_hydra(
        cls,
        config_path: Path | None = None,
        overrides: list[str] | None = None,
    ) -> RegimeSettings:
        path = config_path or _default_config_path()
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "Regime config root must be a mapping",
                code="REGIME_CONFIG_INVALID",
            )
        return cls.from_mapping(container)

    @classmethod
    def default(cls) -> RegimeSettings:
        path = _default_config_path()
        if path.exists():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "regimes" / "default.yaml"
