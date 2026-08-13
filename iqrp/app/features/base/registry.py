"""Dynamic feature registry with automatic registration."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable

from iqrp.app.core.exceptions import ConfigurationError
from iqrp.app.features.base.feature import Feature, FeatureMeta


class FeatureRegistry:
    """Process-local registry of feature classes and instances."""

    def __init__(self) -> None:
        self._classes: dict[str, type[Feature]] = {}
        self._instances: dict[str, Feature] = {}
        self._lock = threading.RLock()

    def register(self, feature_cls: type[Feature]) -> type[Feature]:
        if not hasattr(feature_cls, "meta") or not isinstance(feature_cls.meta, FeatureMeta):
            raise ConfigurationError(
                f"{feature_cls.__name__} must define FeatureMeta on 'meta'",
                code="FEATURE_META_MISSING",
            )
        name = feature_cls.meta.name
        with self._lock:
            self._classes[name] = feature_cls
            self._instances.pop(name, None)
        return feature_cls

    def get_class(self, name: str) -> type[Feature]:
        with self._lock:
            if name not in self._classes:
                raise ConfigurationError(
                    f"Feature '{name}' is not registered",
                    code="FEATURE_NOT_REGISTERED",
                    details={"available": sorted(self._classes)},
                )
            return self._classes[name]

    def get(self, name: str) -> Feature:
        with self._lock:
            if name not in self._instances:
                cls = self.get_class(name)
                self._instances[name] = cls()
            return self._instances[name]

    def list_names(self, *, category: str | None = None) -> list[str]:
        with self._lock:
            names = sorted(self._classes)
        if category is None:
            return names
        return [n for n in names if self.get(n).meta.category == category]

    def describe(self, name: str) -> FeatureMeta:
        return self.get(name).meta

    def dependencies(self, name: str) -> tuple[str, ...]:
        return self.get(name).meta.dependencies

    def all_meta(self) -> list[FeatureMeta]:
        return [self.describe(n) for n in self.list_names()]

    def clear(self) -> None:
        with self._lock:
            self._classes.clear()
            self._instances.clear()


_REGISTRY = FeatureRegistry()


def get_registry() -> FeatureRegistry:
    return _REGISTRY


def register_feature[F: type[Feature]](cls: F) -> F:
    """Class decorator that registers a Feature subclass automatically."""
    _REGISTRY.register(cls)
    return cls


def ensure_features_loaded(modules: Iterable[str] | None = None) -> None:
    """Import feature modules so decorators execute."""
    import importlib

    default = (
        "iqrp.app.features.trend",
        "iqrp.app.features.momentum",
        "iqrp.app.features.volatility",
        "iqrp.app.features.volume",
        "iqrp.app.features.liquidity",
        "iqrp.app.features.microstructure",
        "iqrp.app.features.derivatives",
        "iqrp.app.features.statistical",
        "iqrp.app.features.calendar",
        "iqrp.app.features.cross_asset",
    )
    for mod in modules or default:
        importlib.import_module(mod)


def feature_factory(name: str) -> Callable[[], Feature]:
    cls = get_registry().get_class(name)
    return cls
