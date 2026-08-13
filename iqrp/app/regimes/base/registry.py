"""Automatic registry for regime detection algorithms."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Any

from iqrp.app.core.exceptions import ConfigurationError
from iqrp.app.regimes.base.regime_model import RegimeModel, RegimeModelMeta


class RegimeModelRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type[RegimeModel]] = {}
        self._lock = threading.RLock()

    def register(self, model_cls: type[RegimeModel]) -> type[RegimeModel]:
        if not hasattr(model_cls, "meta") or not isinstance(model_cls.meta, RegimeModelMeta):
            raise ConfigurationError(
                f"{model_cls.__name__} must define RegimeModelMeta on 'meta'",
                code="REGIME_META_MISSING",
            )
        name = model_cls.meta.name
        with self._lock:
            self._classes[name] = model_cls
        return model_cls

    def get_class(self, name: str) -> type[RegimeModel]:
        with self._lock:
            if name not in self._classes:
                raise ConfigurationError(
                    f"Regime model '{name}' is not registered",
                    code="REGIME_MODEL_NOT_REGISTERED",
                    details={"available": sorted(self._classes)},
                )
            return self._classes[name]

    def create(self, name: str, **kwargs: Any) -> RegimeModel:
        cls = self.get_class(name)
        return cls(**kwargs) if kwargs else cls()

    def list_names(self) -> list[str]:
        with self._lock:
            return sorted(self._classes)

    def describe(self, name: str) -> RegimeModelMeta:
        return self.get_class(name).meta

    def all_meta(self) -> list[RegimeModelMeta]:
        return [self.describe(n) for n in self.list_names()]

    def clear(self) -> None:
        with self._lock:
            self._classes.clear()


_REGISTRY = RegimeModelRegistry()


def get_registry() -> RegimeModelRegistry:
    return _REGISTRY


def register_regime_model[M: type[RegimeModel]](cls: M) -> M:
    _REGISTRY.register(cls)
    return cls


def ensure_regime_models_loaded(modules: Iterable[str] | None = None) -> None:
    import importlib

    default = ("iqrp.app.regimes.models.mock",)
    for mod in modules or default:
        importlib.import_module(mod)


def regime_model_factory(name: str) -> Callable[[], RegimeModel]:
    return get_registry().get_class(name)
