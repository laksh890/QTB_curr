"""Hydra-backed configuration for the Institutional Tree-Based Forecasting Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class TaskConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["regression", "binary", "multiclass", "quantile", "probability"] = "regression"
    quantile_alphas: tuple[float, ...] = (0.1, 0.5, 0.9)
    n_classes: int | None = None


class ModelHyperParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = 20
    device: Literal["cpu", "gpu", "cuda"] = "cpu"


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["none", "grid", "random", "bayesian", "optuna"] = "none"
    n_trials: int = 20
    timeout: float | None = None
    parallel: bool = True
    pruning: bool = True
    early_stopping: bool = True
    scoring: str = "neg_rmse"


class ValidationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: Literal[
        "walk_forward",
        "rolling",
        "expanding",
        "blocked",
        "purged_kfold",
        "embargo",
    ] = "walk_forward"
    n_splits: int = 3
    train_size: int = 120
    test_size: int = 24
    gap: int = 0
    embargo: int = 5
    purge: int = 5


class FeatureSelectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    method: Literal[
        "none",
        "rfe",
        "permutation",
        "mutual_info",
        "correlation",
        "shap",
        "boruta",
    ] = "none"
    max_features: int | None = None
    correlation_threshold: float = 0.95


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    method: Literal["none", "platt", "isotonic", "temperature"] = "none"


class RegimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    column: str = "regime"
    mode: Literal["feature", "separate", "weighted", "routing"] = "feature"


class OnlineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["refit", "warm_start", "incremental"] = "warm_start"
    window: int = 500
    refresh_every: int = 50


class EnsembleConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    method: Literal["none", "bagging", "average", "stacking", "blending"] = "none"
    n_bags: int = 5


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


class TreeSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: TaskConfig = Field(default_factory=TaskConfig)
    hyperparameters: ModelHyperParams = Field(default_factory=ModelHyperParams)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    feature_selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    discovery_modules: tuple[str, ...] = (
        "iqrp.app.forecasting.tree_models.xgboost.model",
        "iqrp.app.forecasting.tree_models.lightgbm.model",
        "iqrp.app.forecasting.tree_models.catboost.model",
        "iqrp.app.forecasting.tree_models.sklearn.hist_gradient_boosting",
        "iqrp.app.forecasting.tree_models.sklearn.random_forest",
        "iqrp.app.forecasting.tree_models.sklearn.extra_trees",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> TreeSettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            return cls.model_validate(dict(data or {}))
        except Exception as exc:  # noqa: BLE001
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid tree settings: {exc}",
                code="TREE_CONFIG_INVALID",
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> TreeSettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> TreeSettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "forecasting" / "tree_models" / "default.yaml"
