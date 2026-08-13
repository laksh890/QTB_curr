"""Automatic registry for state-space algorithms."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Any

from iqrp.app.core.exceptions import ConfigurationError
from iqrp.app.state_space.base.state_space_model import StateSpaceModel, StateSpaceModelMeta


class StateSpaceModelRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type[StateSpaceModel]] = {}
        self._lock = threading.RLock()

    def register(self, model_cls: type[StateSpaceModel]) -> type[StateSpaceModel]:
        if not hasattr(model_cls, "meta") or not isinstance(model_cls.meta, StateSpaceModelMeta):
            raise ConfigurationError(
                f"{model_cls.__name__} must define StateSpaceModelMeta on 'meta'",
                code="SS_META_MISSING",
            )
        name = model_cls.meta.name
        with self._lock:
            self._classes[name] = model_cls
        return model_cls

    def get_class(self, name: str) -> type[StateSpaceModel]:
        with self._lock:
            if name not in self._classes:
                raise ConfigurationError(
                    f"State-space model '{name}' is not registered",
                    code="SS_MODEL_NOT_REGISTERED",
                    details={"available": sorted(self._classes)},
                )
            return self._classes[name]

    def create(self, name: str, **kwargs: Any) -> StateSpaceModel:
        cls = self.get_class(name)
        return cls(**kwargs) if kwargs else cls()

    def list_names(self) -> list[str]:
        with self._lock:
            return sorted(self._classes)

    def describe(self, name: str) -> StateSpaceModelMeta:
        return self.get_class(name).meta

    def all_meta(self) -> list[StateSpaceModelMeta]:
        return [self.describe(n) for n in self.list_names()]

    def clear(self) -> None:
        with self._lock:
            self._classes.clear()


_REGISTRY = StateSpaceModelRegistry()


def get_registry() -> StateSpaceModelRegistry:
    return _REGISTRY


def register_state_space_model[M: type[StateSpaceModel]](cls: M) -> M:
    _REGISTRY.register(cls)
    return cls


def ensure_state_space_models_loaded(modules: Iterable[str] | None = None) -> None:
    import importlib

    default = ("iqrp.app.state_space.models.mock",)
    for mod in modules or default:
        importlib.import_module(mod)


def state_space_model_factory(name: str) -> Callable[[], StateSpaceModel]:
    return get_registry().get_class(name)
