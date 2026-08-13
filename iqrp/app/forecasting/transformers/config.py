"""Hydra-backed configuration for Institutional Time-Series Transformer Forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class TaskConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal[
        "regression",
        "classification",
        "binary",
        "multiclass",
        "probability",
        "quantile",
        "distribution",
        "cross_section",
        "multi_target",
    ] = "regression"
    quantile_alphas: tuple[float, ...] = (0.1, 0.5, 0.9)
    n_classes: int = 2
    n_targets: int = 1
    n_assets: int = 1


class ArchitectureConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    lookback: int = 64
    horizon: int = 8
    ffn_dim: int = 128
    patch_len: int = 8
    stride: int = 4
    factor: int = 3
    moving_avg: int = 25
    attention_type: Literal[
        "full", "flash", "sparse", "linear", "performer", "temporal", "cross_asset"
    ] = "full"
    positional: Literal["learned", "sinusoidal", "rotary"] = "sinusoidal"
    use_regime_token: bool = True
    chunk_size: int = 512
    max_context: int = 10240


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: Literal["adamw", "lion", "adam"] = "adamw"
    loss: Literal[
        "mse", "mae", "huber", "quantile", "gaussian_nll", "student_t_nll", "bce", "cross_entropy"
    ] = "mse"
    grad_clip: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = False
    compile: bool = False
    gradient_checkpointing: bool = False
    early_stopping_patience: int = 8
    seed: int = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    val_ratio: float = 0.2
    teacher_forcing: float = 0.5
    scheduled_sampling: bool = False
    curriculum: bool = False
    ema_decay: float = 0.0
    label_smoothing: float = 0.0


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["none", "cosine", "warmup_cosine"] = "warmup_cosine"
    warmup_epochs: int = 2
    min_lr: float = 1e-6


class RegimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    column: str = "regime"
    mode: Literal["token", "embedding", "mask", "experts", "feature"] = "embedding"
    n_regimes: int = 4


class OnlineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["refit", "finetune", "warm_start"] = "finetune"
    window: int = 1024
    refresh_every: int = 50
    finetune_epochs: int = 2


class ProbabilisticConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    distribution: Literal["gaussian", "student_t", "quantile", "mixture"] = "gaussian"
    n_mixtures: int = 3
    n_samples: int = 50


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    target: str = "target"
    asset_column: str | None = None
    feature_columns: tuple[str, ...] | None = None


class ForecastConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_horizon: int = 8
    interval_level: float = 0.95
    multi_horizon: bool = True
    streaming: bool = False
    sliding_context: bool = True


class DistributedConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    ddp: bool = False
    amp: bool = False


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_points: int = 500


class TransformerSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: TaskConfig = Field(default_factory=TaskConfig)
    architecture: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    probabilistic: ProbabilisticConfig = Field(default_factory=ProbabilisticConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    discovery_modules: tuple[str, ...] = (
        "iqrp.app.forecasting.transformers.architectures.tft.model",
        "iqrp.app.forecasting.transformers.architectures.informer.model",
        "iqrp.app.forecasting.transformers.architectures.autoformer.model",
        "iqrp.app.forecasting.transformers.architectures.fedformer.model",
        "iqrp.app.forecasting.transformers.architectures.patchtst.model",
        "iqrp.app.forecasting.transformers.architectures.crossformer.model",
        "iqrp.app.forecasting.transformers.architectures.timesnet.model",
        "iqrp.app.forecasting.transformers.architectures.itransformer.model",
        "iqrp.app.forecasting.transformers.architectures.timemixer.model",
        "iqrp.app.forecasting.transformers.architectures.tide.model",
        "iqrp.app.forecasting.transformers.architectures.moe_transformer.model",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> TransformerSettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            return cls.model_validate(dict(data or {}))
        except Exception as exc:  # noqa: BLE001
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid transformer settings: {exc}",
                code="TRANSFORMER_CONFIG_INVALID",
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> TransformerSettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> TransformerSettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "forecasting" / "transformers" / "default.yaml"
