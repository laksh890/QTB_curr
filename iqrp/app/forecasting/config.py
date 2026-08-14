"""Hydra-backed configuration for the Institutional Forecasting Framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    target: str | None = "target"
    feature_columns: tuple[str, ...] | None = None
    regime_column: str | None = "regime"


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    scaler: Literal["none", "standard", "minmax", "robust"] = "standard"
    encode_categoricals: bool = True
    window_size: int = 32
    horizon: int = 1
    feature_selection: Literal["none", "variance", "correlation", "mutual_info"] = "none"
    max_features: int | None = None
    variance_threshold: float = 0.0
    correlation_threshold: float = 0.95


class PostprocessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    calibrate: bool = False
    calibration_method: Literal["platt", "isotonic", "temperature", "none"] = "none"
    interval_level: float = 0.95
    interval_method: Literal["residual", "quantile", "gaussian"] = "residual"
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


class OnlineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    warm_start: bool = True
    rolling_retrain_every: int = 0
    checkpoint_every: int = 0
    stream_buffer: int = 256


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    validation_fraction: float = 0.2
    validation_method: Literal[
        "holdout", "cross_validation", "walk_forward", "rolling", "time_series_split"
    ] = "holdout"
    n_splits: int = 5
    walk_forward_train_size: int | None = None
    walk_forward_test_size: int = 1
    rolling_window: int | None = None


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    primary_metric: str = "rmse"
    include_financial: bool = True
    include_classification: bool = True
    include_probability: bool = True


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_points: int = 500


class SerializationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    format: Literal["json"] = "json"
    include_npz: bool = True


class InferenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_horizon: int = 5
    strategy: Literal["direct", "recursive", "sequence", "multi_step"] = "direct"
    batch_size: int = 1024


class ForecastingSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    postprocessing: PostprocessingConfig = Field(default_factory=PostprocessingConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    discovery_modules: tuple[str, ...] = ("iqrp.app.forecasting.models.mock",)

    @classmethod
    def from_mapping(cls, data: Any) -> ForecastingSettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            return cls.model_validate(dict(data or {}))
        except Exception as exc:
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid forecasting settings: {exc}",
                code="FC_CONFIG_INVALID",
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> ForecastingSettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> ForecastingSettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "forecasting" / "default.yaml"
