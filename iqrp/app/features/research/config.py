"""Hydra / OmegaConf configuration for the feature research engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    close: str = "close"
    feature_prefix: str | None = None


class TargetsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    return_horizon: int = 1
    volatility_window: int = 20
    drawdown_window: int = 20
    direction_threshold: float = 0.0
    regime_vol_window: int = 50
    regime_quantiles: tuple[float, ...] = (0.33, 0.66)


class StatisticsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    entropy_bins: int = 20
    jarque_bera_alpha: float = 0.05


class CorrelationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    pearson: bool = True
    spearman: bool = True
    kendall: bool = True
    distance: bool = True
    mutual_information: bool = True
    mic: bool = True
    cross_correlation_max_lag: int = 5
    rolling_window: int = 60
    high_correlation_threshold: float = 0.9
    clustering_linkage: str = "average"
    clustering_distance_threshold: float = 0.3
    mi_bins: int = 16


class RedundancyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    duplicate_atol: float = 1e-12
    near_duplicate_threshold: float = 0.995
    vif_threshold: float = 10.0
    linear_dependence_rank_tol: float = 1e-8
    rolling_window_name_pattern: str = r"_(?:roll|rolling|sma|ema|window)\d+"


class PredictiveConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizons: tuple[int, ...] = (1, 5, 10)
    min_train_size: int = 60
    test_size: int = 20
    step_size: int = 20
    blocked_n_splits: int = 5
    blocked_purge: int = 5
    evaluation_mode: Literal["rolling", "expanding", "walk_forward", "blocked"] = "walk_forward"
    classification_threshold: float = 0.0
    mi_bins: int = 16


class StabilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    rolling_window: int = 60
    step: int = 20
    ic_min_obs: int = 30
    decay_half_life_bars: int = 100
    parameter_drift_window: int = 80


class DriftConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    reference_fraction: float = 0.3
    psi_bins: int = 10
    psi_alert_threshold: float = 0.2
    ks_alert_threshold: float = 0.1
    mean_shift_z_threshold: float = 3.0
    concept_ic_drop_threshold: float = 0.5


class ImportanceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_permutations: int = 20
    rfe_n_features_to_select: int = 5
    sfs_n_features_to_select: int = 5
    shap_enabled: bool = True
    model: Literal["ridge", "stump"] = "ridge"
    ridge_alpha: float = 1.0


class ScoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    weight_predictive_power: float = 0.35
    weight_stability: float = 0.20
    weight_redundancy_penalty: float = 0.15
    weight_computational_cost: float = 0.05
    weight_interpretability: float = 0.10
    weight_consistency_assets: float = 0.075
    weight_consistency_timeframes: float = 0.075
    accept_score_threshold: float = 60.0
    reject_score_threshold: float = 35.0
    weak_score_threshold: float = 50.0


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    format: Literal["svg", "html"] = "svg"
    dpi: int = 120
    max_features_in_charts: int = 40


class ReportsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    write_markdown: bool = True
    write_json: bool = True
    include_charts: bool = True


class ResearchSettings(BaseModel):
    """Immutable research-engine configuration (Hydra-backed)."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    n_jobs: int = 4
    random_seed: int = 42
    cache_enabled: bool = True
    cache_dir: str = "data/cache/feature_research"
    output_dir: str = "data/reports/feature_research"
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    statistics: StatisticsConfig = Field(default_factory=StatisticsConfig)
    correlation: CorrelationConfig = Field(default_factory=CorrelationConfig)
    redundancy: RedundancyConfig = Field(default_factory=RedundancyConfig)
    predictive: PredictiveConfig = Field(default_factory=PredictiveConfig)
    stability: StabilityConfig = Field(default_factory=StabilityConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    importance: ImportanceConfig = Field(default_factory=ImportanceConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    reports: ReportsConfig = Field(default_factory=ReportsConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | Any) -> ResearchSettings:
        from iqrp.app.core.exceptions import ConfigurationError

        if not isinstance(data, dict):
            if OmegaConf.is_config(data):
                container = OmegaConf.to_container(data, resolve=True)
            elif hasattr(data, "items"):
                container = dict(data.items())
            else:
                container = data
            if not isinstance(container, dict):
                raise ConfigurationError(
                    "Research config mapping invalid",
                    code="RESEARCH_CONFIG_INVALID",
                )
            data = container
        return cls.model_validate(data)

    @classmethod
    def from_hydra(
        cls,
        config_path: Path | None = None,
        overrides: list[str] | None = None,
    ) -> ResearchSettings:
        """Load ``configs/research/default.yaml`` with optional OmegaConf overrides."""
        path = config_path or _default_config_path()
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "Research config root must be a mapping",
                code="RESEARCH_CONFIG_INVALID",
            )
        return cls.from_mapping(container)

    @classmethod
    def default(cls) -> ResearchSettings:
        path = _default_config_path()
        if path.exists():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    # iqrp/app/features/research/config.py -> repo/iqrp/configs/research/default.yaml
    return Path(__file__).resolve().parents[3] / "configs" / "research" / "default.yaml"
