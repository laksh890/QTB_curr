"""Hydra-backed configuration for the Gaussian Mixture Regime Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class CovarianceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["full", "diag", "tied", "spherical"] = "full"
    reg_covar: float = 1.0e-6


class InitializationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["random", "kmeans", "kmeans++", "hierarchical", "user"] = "kmeans"
    n_restarts: int = 3


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_iter: int = 100
    tol: float = 1.0e-4
    early_stopping: bool = True
    n_jobs: int = 2
    warm_start: bool = False


class BayesianConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    weight_concentration_prior: float = 1.0
    mean_precision_prior: float = 1.0
    degrees_of_freedom_prior: float | None = None
    covariance_prior_scale: float = 1.0


class PreprocessingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    standardize: bool = True
    whiten: bool = False
    pca_components: int | None = None
    ica_components: int | None = None


class OnlineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    window_size: int = 0
    update_frequency: int = 1
    warm_start: bool = True
    adaptive_covariance: bool = True


class ModelSelectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_components: int = 1
    max_components: int = 5
    criterion: Literal["aic", "bic", "icl", "log_likelihood", "cv"] = "bic"
    cv_folds: int = 3


class OutlierConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    density_quantile: float = 0.01
    rare_occupancy: float = 0.05


class ForecastingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_horizon: int = 5
    confidence_level: float = 0.95


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    observation_columns: tuple[str, ...] | None = None


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_points: int = 500


class SerializationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    format: Literal["json"] = "json"


class GMMSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    n_components: int = 3
    n_features: int = 1
    state_names: tuple[str, ...] | None = None
    random_seed: int = 42
    store_dir: str = "data/gmm"
    output_dir: str = "data/reports/gmm"
    model_type: Literal["gmm", "bayesian_gmm"] = "gmm"
    covariance: CovarianceConfig = Field(default_factory=CovarianceConfig)
    initialization: InitializationConfig = Field(default_factory=InitializationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    bayesian: BayesianConfig = Field(default_factory=BayesianConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    model_selection: ModelSelectionConfig = Field(default_factory=ModelSelectionConfig)
    outlier: OutlierConfig = Field(default_factory=OutlierConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    serialization: SerializationConfig = Field(default_factory=SerializationConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | Any) -> GMMSettings:
        from iqrp.app.core.exceptions import ConfigurationError

        if not isinstance(data, dict):
            if OmegaConf.is_config(data):
                container = OmegaConf.to_container(data, resolve=True)
            else:
                container = data
            if not isinstance(container, dict):
                raise ConfigurationError("GMM config mapping invalid", code="GMM_CONFIG_INVALID")
            data = container
        return cls.model_validate(data)

    @classmethod
    def from_hydra(
        cls,
        config_path: Path | None = None,
        overrides: list[str] | None = None,
    ) -> GMMSettings:
        path = config_path or _default_config_path()
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError("GMM config root must be a mapping", code="GMM_CONFIG_INVALID")
        return cls.from_mapping(container)

    @classmethod
    def default(cls) -> GMMSettings:
        path = _default_config_path()
        if path.exists():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "gmm" / "default.yaml"
