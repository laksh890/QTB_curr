"""Hydra-backed configuration for the Markov Chain Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class EstimationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["mle", "bayesian", "frequency", "weighted"] = "bayesian"
    laplace_alpha: float = 1.0
    dirichlet_alpha: float = 1.0
    forgetting_factor: float = 1.0
    window_size: int = 0
    min_count_warning: int = 5


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    state_column: str = "state_id"
    weight_column: str | None = None


class ForecastingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_horizon: int = 5
    confidence_level: float = 0.95


class OnlineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    update_frequency: int = 1
    adaptive: bool = True


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_points: int = 500


class SerializationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    format: Literal["json"] = "json"


class MarkovSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    n_states: int = 3
    state_names: tuple[str, ...] | None = None
    random_seed: int = 42
    store_dir: str = "data/markov"
    output_dir: str = "data/reports/markov"
    estimation: EstimationConfig = Field(default_factory=EstimationConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | Any) -> MarkovSettings:
        from iqrp.app.core.exceptions import ConfigurationError

        if not isinstance(data, dict):
            if OmegaConf.is_config(data):
                container = OmegaConf.to_container(data, resolve=True)
            else:
                container = data
            if not isinstance(container, dict):
                raise ConfigurationError(
                    "Markov config mapping invalid",
                    code="MARKOV_CONFIG_INVALID",
                )
            data = container
        return cls.model_validate(data)

    @classmethod
    def from_hydra(
        cls,
        config_path: Path | None = None,
        overrides: list[str] | None = None,
    ) -> MarkovSettings:
        path = config_path or _default_config_path()
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "Markov config root must be a mapping",
                code="MARKOV_CONFIG_INVALID",
            )
        return cls.from_mapping(container)

    @classmethod
    def default(cls) -> MarkovSettings:
        path = _default_config_path()
        if path.exists():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "markov" / "default.yaml"
