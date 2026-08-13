"""Dynamic label registry with automatic registration."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable

from iqrp.app.core.exceptions import ConfigurationError
from iqrp.app.labels.base.label import Label, LabelMeta


class LabelRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type[Label]] = {}
        self._instances: dict[str, Label] = {}
        self._lock = threading.RLock()

    def register(self, label_cls: type[Label]) -> type[Label]:
        if not hasattr(label_cls, "meta") or not isinstance(label_cls.meta, LabelMeta):
            raise ConfigurationError(
                f"{label_cls.__name__} must define LabelMeta on 'meta'",
                code="LABEL_META_MISSING",
            )
        name = label_cls.meta.name
        with self._lock:
            self._classes[name] = label_cls
            self._instances.pop(name, None)
        return label_cls

    def get_class(self, name: str) -> type[Label]:
        with self._lock:
            if name not in self._classes:
                raise ConfigurationError(
                    f"Label '{name}' is not registered",
                    code="LABEL_NOT_REGISTERED",
                    details={"available": sorted(self._classes)},
                )
            return self._classes[name]

    def get(self, name: str) -> Label:
        with self._lock:
            if name not in self._instances:
                self._instances[name] = self.get_class(name)()
            return self._instances[name]

    def list_names(self, *, category: str | None = None) -> list[str]:
        with self._lock:
            names = sorted(self._classes)
        if category is None:
            return names
        return [n for n in names if self.get(n).meta.category == category]

    def describe(self, name: str) -> LabelMeta:
        return self.get(name).meta

    def dependencies(self, name: str) -> tuple[str, ...]:
        return self.get(name).meta.dependencies

    def all_meta(self) -> list[LabelMeta]:
        return [self.describe(n) for n in self.list_names()]

    def clear(self) -> None:
        with self._lock:
            self._classes.clear()
            self._instances.clear()


_REGISTRY = LabelRegistry()


def get_registry() -> LabelRegistry:
    return _REGISTRY


def register_label[L: type[Label]](cls: L) -> L:
    _REGISTRY.register(cls)
    return cls


def ensure_labels_loaded(modules: Iterable[str] | None = None) -> None:
    import importlib

    default = (
        "iqrp.app.labels.regression",
        "iqrp.app.labels.classification",
        "iqrp.app.labels.survival",
        "iqrp.app.labels.volatility",
        "iqrp.app.labels.regime",
        "iqrp.app.labels.barrier",
        "iqrp.app.labels.meta",
    )
    for mod in modules or default:
        importlib.import_module(mod)


def label_factory(name: str) -> Callable[[], Label]:
    return get_registry().get_class(name)
