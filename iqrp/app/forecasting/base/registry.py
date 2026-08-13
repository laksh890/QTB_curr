"""Automatic registry for forecasting algorithms."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Iterable
from typing import Any, TYPE_CHECKING

from iqrp.app.core.exceptions import ConfigurationError
from iqrp.app.forecasting.base.metadata import ForecastModelMeta, TrainingMetadata

if TYPE_CHECKING:
    from iqrp.app.forecasting.base.forecast_model import ForecastModel


class ForecastModelRegistry:
    """Process-wide registry of forecasting model classes and training records."""

    def __init__(self) -> None:
        self._classes: dict[str, type[ForecastModel]] = {}
        self._training: dict[str, list[TrainingMetadata]] = {}
        self._configs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def register(self, model_cls: type[ForecastModel]) -> type[ForecastModel]:
        if not hasattr(model_cls, "meta") or not isinstance(model_cls.meta, ForecastModelMeta):
            raise ConfigurationError(
                f"{model_cls.__name__} must define ForecastModelMeta on 'meta'",
                code="FC_META_MISSING",
            )
        name = model_cls.meta.name
        with self._lock:
            self._classes[name] = model_cls
            self._configs.setdefault(name, dict(model_cls.meta.parameters))
        return model_cls

    def get_class(self, name: str) -> type[ForecastModel]:
        with self._lock:
            if name not in self._classes:
                raise ConfigurationError(
                    f"Forecast model '{name}' is not registered",
                    code="FC_MODEL_NOT_REGISTERED",
                    details={"available": sorted(self._classes)},
                )
            return self._classes[name]

    def create(self, name: str, **kwargs: Any) -> ForecastModel:
        cls = self.get_class(name)
        return cls(**kwargs) if kwargs else cls()

    def list_names(self) -> list[str]:
        with self._lock:
            return sorted(self._classes)

    def describe(self, name: str) -> ForecastModelMeta:
        return self.get_class(name).meta

    def all_meta(self) -> list[ForecastModelMeta]:
        return [self.describe(n) for n in self.list_names()]

    def record_training(self, name: str, meta: TrainingMetadata) -> None:
        with self._lock:
            self._training.setdefault(name, []).append(meta)

    def training_history(self, name: str) -> list[TrainingMetadata]:
        with self._lock:
            return list(self._training.get(name, []))

    def set_config(self, name: str, config: dict[str, Any]) -> None:
        with self._lock:
            self._configs[name] = dict(config)

    def get_config(self, name: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._configs.get(name, {}))

    def clear(self) -> None:
        with self._lock:
            self._classes.clear()
            self._training.clear()
            self._configs.clear()


_REGISTRY = ForecastModelRegistry()


def get_registry() -> ForecastModelRegistry:
    return _REGISTRY


def register_forecast_model[M: type[ForecastModel]](cls: M) -> M:
    _REGISTRY.register(cls)
    return cls


def ensure_forecast_models_loaded(modules: Iterable[str] | None = None) -> None:
    default = ("iqrp.app.forecasting.models.mock",)
    for mod in modules or default:
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001 - optional until algorithms land
            continue


def forecast_model_factory(name: str) -> Callable[[], ForecastModel]:
    return get_registry().get_class(name)
