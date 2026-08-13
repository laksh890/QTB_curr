"""Path generator contract and registry for stochastic models."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.core.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class GeneratorMeta:
    name: str
    version: str
    description: str
    family: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "family": self.family,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class PathResult:
    """Generated price path with optional latent state (e.g. variance)."""

    prices: np.ndarray  # shape (T+1,) or (T+1, n_assets)
    returns: np.ndarray  # shape (T,) or (T, n_assets)
    volatility: np.ndarray  # instantaneous / realized vol path
    drift: np.ndarray  # per-step drift used
    latent: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class PathGenerator(ABC):
    """Interchangeable stochastic path generator."""

    meta: GeneratorMeta

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    @abstractmethod
    def generate(
        self,
        n_steps: int,
        *,
        x0: float | np.ndarray = 100.0,
        dt: float = 0.004,
        **params: Any,
    ) -> PathResult:
        """Simulate a path of length ``n_steps`` (returns ``n_steps`` increments)."""


class GeneratorRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type[PathGenerator]] = {}
        self._lock = threading.RLock()

    def register(self, cls: type[PathGenerator]) -> type[PathGenerator]:
        if not hasattr(cls, "meta") or not isinstance(cls.meta, GeneratorMeta):
            raise ConfigurationError(
                f"{cls.__name__} must define GeneratorMeta on 'meta'",
                code="SIM_META_MISSING",
            )
        with self._lock:
            self._classes[cls.meta.name] = cls
        return cls

    def get_class(self, name: str) -> type[PathGenerator]:
        with self._lock:
            if name not in self._classes:
                raise ConfigurationError(
                    f"Path generator '{name}' is not registered",
                    code="SIM_GENERATOR_NOT_REGISTERED",
                    details={"available": sorted(self._classes)},
                )
            return self._classes[name]

    def create(
        self, name: str, rng: np.random.Generator | None = None, **kwargs: Any
    ) -> PathGenerator:
        cls = self.get_class(name)
        return cls(rng=rng, **kwargs)

    def list_names(self) -> list[str]:
        with self._lock:
            return sorted(self._classes)

    def describe(self, name: str) -> GeneratorMeta:
        return self.get_class(name).meta

    def clear(self) -> None:
        with self._lock:
            self._classes.clear()


_REGISTRY = GeneratorRegistry()


def get_generator_registry() -> GeneratorRegistry:
    return _REGISTRY


def register_generator[G: type[PathGenerator]](cls: G) -> G:
    _REGISTRY.register(cls)
    return cls


def ensure_generators_loaded(modules: tuple[str, ...] | None = None) -> None:
    import importlib

    default = (
        "iqrp.app.simulation.stochastic.gbm",
        "iqrp.app.simulation.stochastic.ou",
        "iqrp.app.simulation.stochastic.jump_diffusion",
        "iqrp.app.simulation.stochastic.heston",
        "iqrp.app.simulation.stochastic.variance_gamma",
        "iqrp.app.simulation.stochastic.cir",
        "iqrp.app.simulation.stochastic.random_walk",
    )
    for mod in modules or default:
        importlib.import_module(mod)


def generator_factory(name: str) -> Callable[[], type[PathGenerator]]:
    return lambda: get_generator_registry().get_class(name)
