"""Hydra-backed configuration for the Ensemble Regime Intelligence Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class CombinationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal[
        "majority",
        "weighted",
        "soft_voting",
        "bma",
        "stacking",
        "confidence",
        "dynamic",
        "meta",
    ] = "soft_voting"
    normalize: bool = True


class WeightingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal[
        "equal",
        "accuracy",
        "recent_accuracy",
        "log_likelihood",
        "calibration",
        "stability",
        "user",
        "rolling",
        "adaptive",
    ] = "equal"
    user_weights: dict[str, float] | None = None
    lookback: int = 50
    adaptive_rate: float = 0.05
    min_weight: float = 0.01


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    method: Literal["platt", "isotonic", "temperature", "dirichlet", "none"] = "temperature"
    temperature: float = 1.0


class OnlineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    warm_start: bool = True
    weight_update: bool = True
    recalibrate_every: int = 0


class ForecastingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_horizon: int = 5
    confidence_level: float = 0.95


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    feature_columns: tuple[str, ...] | None = None


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_points: int = 500


class SerializationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    format: Literal["json"] = "json"


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    validation_fraction: float = 0.25
    min_members: int = 1


class EnsembleSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    n_states: int = 6
    state_names: tuple[str, ...] = (
        "bull",
        "bear",
        "sideways",
        "high_volatility",
        "low_volatility",
        "liquidity_stress",
    )
    random_seed: int = 42
    store_dir: str = "data/ensemble"
    output_dir: str = "data/reports/ensemble"
    discovery_modules: tuple[str, ...] = ("iqrp.app.regimes.models.mock",)
    member_names: tuple[str, ...] | None = None
    combination: CombinationConfig = Field(default_factory=CombinationConfig)
    weighting: WeightingConfig = Field(default_factory=WeightingConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | Any) -> EnsembleSettings:
        from iqrp.app.core.exceptions import ConfigurationError

        if not isinstance(data, dict):
            if OmegaConf.is_config(data):
                container = OmegaConf.to_container(data, resolve=True)
            else:
                container = data
            if not isinstance(container, dict):
                raise ConfigurationError(
                    "Ensemble config mapping invalid", code="ENS_CONFIG_INVALID"
                )
            data = container
        return cls.model_validate(data)

    @classmethod
    def from_hydra(
        cls,
        config_path: Path | None = None,
        overrides: list[str] | None = None,
    ) -> EnsembleSettings:
        path = config_path or _default_config_path()
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "Ensemble config root must be a mapping", code="ENS_CONFIG_INVALID"
            )
        return cls.from_mapping(container)

    @classmethod
    def default(cls) -> EnsembleSettings:
        path = _default_config_path()
        if path.exists():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "ensemble" / "default.yaml"
