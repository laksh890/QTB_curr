"""Hydra-backed configuration for the Institutional Neural Forecasting Platform."""

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
        "sequence",
    ] = "regression"
    quantile_alphas: tuple[float, ...] = (0.1, 0.5, 0.9)
    n_classes: int = 2
    n_targets: int = 1


class ArchitectureConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False
    lookback: int = 32
    horizon: int = 5
    kernel_size: int = 3
    n_blocks: int = 2
    n_harmonics: int = 4
    n_polynomials: int = 2
    attention: bool = True
    embedding_dim: int = 16


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: Literal["adam", "adamw", "rmsprop", "sgd", "lion", "lookahead"] = "adamw"
    loss: Literal[
        "mse",
        "mae",
        "huber",
        "logcosh",
        "cross_entropy",
        "bce",
        "focal",
        "quantile",
        "gaussian_nll",
        "student_t_nll",
    ] = "mse"
    grad_clip: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = False
    compile: bool = False
    early_stopping_patience: int = 8
    seed: int = 42
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    num_workers: int = 0
    val_ratio: float = 0.2
    label_smoothing: float = 0.0


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["none", "cosine", "onecycle", "plateau", "warmup_cosine", "exponential"] = "cosine"
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    gamma: float = 0.95


class RegimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    column: str = "regime"
    mode: Literal["feature", "embedding", "gating", "separate", "moe"] = "feature"
    n_regimes: int = 4


class OnlineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["refit", "finetune", "warm_start"] = "finetune"
    window: int = 512
    refresh_every: int = 50
    finetune_epochs: int = 3


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["none", "grid", "random", "bayesian", "optuna"] = "none"
    n_trials: int = 10
    pruning: bool = True
    parallel: bool = False


class ProbabilisticConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    distribution: Literal["gaussian", "student_t", "quantile"] = "gaussian"
    n_samples: int = 50
    mc_dropout: bool = False


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    target: str = "target"
    feature_columns: tuple[str, ...] | None = None


class ForecastConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_horizon: int = 5
    interval_level: float = 0.95
    multi_horizon: bool = True


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_points: int = 500


class DistributedConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    ddp: bool = False
    amp: bool = False
    gradient_checkpointing: bool = False


class NeuralSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: TaskConfig = Field(default_factory=TaskConfig)
    architecture: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    probabilistic: ProbabilisticConfig = Field(default_factory=ProbabilisticConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    discovery_modules: tuple[str, ...] = (
        "iqrp.app.forecasting.neural.mlp.model",
        "iqrp.app.forecasting.neural.lstm.model",
        "iqrp.app.forecasting.neural.gru.model",
        "iqrp.app.forecasting.neural.tcn.model",
        "iqrp.app.forecasting.neural.nbeats.model",
        "iqrp.app.forecasting.neural.nhits.model",
        "iqrp.app.forecasting.neural.deepar.model",
        "iqrp.app.forecasting.neural.seq2seq.model",
        "iqrp.app.forecasting.neural.variants",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> NeuralSettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            return cls.model_validate(dict(data or {}))
        except Exception as exc:  # noqa: BLE001
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid neural settings: {exc}",
                code="NEURAL_CONFIG_INVALID",
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> NeuralSettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> NeuralSettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "forecasting" / "neural" / "default.yaml"
