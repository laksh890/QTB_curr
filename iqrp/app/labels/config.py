"""Hydra-backed configuration for the Label Engineering Platform."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"


class DefaultsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizon: int = 12
    volatility_window: int = 20
    atr_window: int = 14
    return_threshold: float = 0.0
    bucket_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
    stress_vol_quantile: float = 0.9


class TripleBarrierConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizon: int = 20
    upper_mult: float = 1.0
    lower_mult: float = 1.0
    barrier_mode: Literal["fixed", "atr", "volatility"] = "atr"
    fixed_upper: float = 0.02
    fixed_lower: float = 0.02
    atr_window: int = 14
    atr_multiplier: float = 2.0
    vol_window: int = 20
    vol_multiplier: float = 2.0
    min_return: float = 0.0


class MetaLabelingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    primary_signal_column: str = "primary_signal"
    confirmation_column: str | None = None
    probability_threshold: float = 0.5
    side_column: str | None = None


class RegimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    trend_window: int = 50
    vol_window: int = 30
    liquidity_window: int = 30
    sideways_threshold: float = 0.002
    bull_threshold: float = 0.01
    bear_threshold: float = -0.01
    vol_quantiles: tuple[float, ...] = (0.33, 0.66)
    liquidity_quantiles: tuple[float, ...] = (0.33, 0.66)


class ValidationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_lookahead_tolerance: int = 0
    imbalance_ratio_alert: float = 0.05
    min_coverage: float = 0.5
    entropy_bins: int = 20


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_points: int = 500


class ReportsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    write_markdown: bool = True
    write_json: bool = True


class LabelSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    n_jobs: int = 4
    random_seed: int = 42
    output_dir: str = "data/reports/labels"
    store_dir: str = "data/labels"
    cache_enabled: bool = True
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    triple_barrier: TripleBarrierConfig = Field(default_factory=TripleBarrierConfig)
    meta_labeling: MetaLabelingConfig = Field(default_factory=MetaLabelingConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    reports: ReportsConfig = Field(default_factory=ReportsConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | Any) -> LabelSettings:
        from iqrp.app.core.exceptions import ConfigurationError

        if not isinstance(data, dict):
            if OmegaConf.is_config(data):
                container = OmegaConf.to_container(data, resolve=True)
            else:
                container = data
            if not isinstance(container, dict):
                raise ConfigurationError(
                    "Label config mapping invalid",
                    code="LABEL_CONFIG_INVALID",
                )
            data = container
        return cls.model_validate(data)

    @classmethod
    def from_hydra(
        cls,
        config_path: Path | None = None,
        overrides: list[str] | None = None,
    ) -> LabelSettings:
        path = config_path or _default_config_path()
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "Label config root must be a mapping",
                code="LABEL_CONFIG_INVALID",
            )
        return cls.from_mapping(container)

    @classmethod
    def default(cls) -> LabelSettings:
        path = _default_config_path()
        if path.exists():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "labels" / "default.yaml"
