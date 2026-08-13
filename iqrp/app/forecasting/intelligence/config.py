"""Hydra-backed configuration for Institutional Forecast Intelligence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    root_package: str = "iqrp.app.forecasting"
    include_families: tuple[str, ...] | None = None
    exclude_families: tuple[str, ...] = ()
    exclude_names: tuple[str, ...] = ()
    max_candidates: int | None = None


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal[
        "walk_forward",
        "rolling",
        "time_series_split",
        "nested_cv",
        "purged_kfold",
        "embargo",
    ] = "walk_forward"
    n_splits: int = 3
    train_size: int = 80
    test_size: int = 20
    gap: int = 0
    embargo: int = 5
    purge: int = 5
    parallel: bool = True
    max_workers: int = 4


class RankingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    primary: str = "rmse"
    higher_is_better: tuple[str, ...] = (
        "directional_accuracy",
        "sharpe",
        "sortino",
        "profit_factor",
        "r2",
    )
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "rmse": 1.0,
            "mae": 0.5,
            "directional_accuracy": 0.8,
            "sharpe": 0.3,
        }
    )


class AutoMLConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal[
        "none",
        "random",
        "grid",
        "bayesian",
        "optuna",
        "hyperband",
        "successive_halving",
        "pbt",
    ] = "none"
    n_trials: int = 10
    timeout: float | None = None
    multi_objective: bool = False
    objectives: tuple[str, ...] = ("rmse", "directional_accuracy")


class EnsembleConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal[
        "none",
        "weighted",
        "median",
        "bma",
        "stacking",
        "blending",
        "voting",
        "moe",
        "dynamic",
    ] = "weighted"
    top_k: int = 3
    min_weight: float = 0.05


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    method: Literal["none", "temperature", "platt", "isotonic", "dirichlet"] = "none"


class DriftConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    feature_psi_threshold: float = 0.2
    prediction_ks_threshold: float = 0.1
    performance_drop: float = 0.25


class RetrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["scheduled", "performance", "drift", "rolling", "none"] = "performance"
    schedule_every: int = 100
    window: int = 500
    warm_start: bool = True


class RoutingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    by_regime: bool = True
    by_volatility: bool = True
    by_confidence: bool = True
    vol_column: str = "vol_forecast"
    regime_column: str = "regime"


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    window: int = 200
    latency_budget_ms: float = 100.0
    accuracy_floor: float = 0.0
    mae_alert: float = 1.0
    latency_ms_alert: float = 100.0
    calibration_alert: float = 0.25
    stability_alert: float = 0.2


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    target: str = "target"
    feature_columns: tuple[str, ...] | None = None


class ForecastConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_horizon: int = 5
    interval_level: float = 0.95


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    top_n: int = 10


class IntelligenceSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    automl: AutoMLConfig = Field(default_factory=AutoMLConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    retrain: RetrainConfig = Field(default_factory=RetrainConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    seed: int = 42

    @classmethod
    def from_mapping(cls, data: Any) -> IntelligenceSettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            return cls.model_validate(dict(data or {}))
        except Exception as exc:  # noqa: BLE001
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid intelligence settings: {exc}",
                code="FI_CONFIG_INVALID",
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> IntelligenceSettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> IntelligenceSettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "forecasting" / "intelligence" / "default.yaml"
