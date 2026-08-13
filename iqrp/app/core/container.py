"""Lightweight dependency-injection container.

Providers are registered by type (or an explicit string key) and resolved
lazily. No global mutable application state is held outside this container;
call :func:`reset_container` between tests to isolate fixtures.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, TypeVar, cast

from iqrp.app.core.exceptions import ConfigurationError

T = TypeVar("T")
Provider = Callable[[], Any]


class Container:
    """Thread-safe service registry with singleton and factory scopes."""

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._singletons: dict[str, Any] = {}
        self._singleton_keys: set[str] = set()
        self._lock = threading.RLock()

    def register(
        self,
        key: type[Any] | str,
        provider: Provider,
        *,
        singleton: bool = True,
    ) -> None:
        """Register a provider for ``key``.

        Args:
            key: Concrete type or string identifier.
            provider: Zero-arg callable that constructs the dependency.
            singleton: If True, the first resolution is cached.
        """
        name = self._key_name(key)
        with self._lock:
            self._providers[name] = provider
            if singleton:
                self._singleton_keys.add(name)
            else:
                self._singleton_keys.discard(name)
                self._singletons.pop(name, None)

    def register_instance(self, key: type[Any] | str, instance: Any) -> None:
        """Register an already-constructed instance as a singleton."""
        name = self._key_name(key)
        with self._lock:
            self._providers[name] = lambda: instance
            self._singleton_keys.add(name)
            self._singletons[name] = instance

    def resolve(self, key: type[T] | str) -> T:
        """Resolve a dependency, constructing it if necessary."""
        name = self._key_name(key)
        with self._lock:
            if name in self._singletons:
                return cast(T, self._singletons[name])
            if name not in self._providers:
                raise ConfigurationError(
                    f"No provider registered for '{name}'",
                    code="DI_PROVIDER_MISSING",
                    details={"key": name},
                )
            instance = self._providers[name]()
            if name in self._singleton_keys:
                self._singletons[name] = instance
            return cast(T, instance)

    def has(self, key: type[Any] | str) -> bool:
        """Return True if a provider (or instance) exists for ``key``."""
        return self._key_name(key) in self._providers

    def unregister(self, key: type[Any] | str) -> None:
        """Remove a provider and any cached singleton."""
        name = self._key_name(key)
        with self._lock:
            self._providers.pop(name, None)
            self._singletons.pop(name, None)
            self._singleton_keys.discard(name)

    def clear(self) -> None:
        """Remove all providers and cached instances."""
        with self._lock:
            self._providers.clear()
            self._singletons.clear()
            self._singleton_keys.clear()

    @staticmethod
    def _key_name(key: type[Any] | str) -> str:
        if isinstance(key, str):
            return key
        return f"{key.__module__}.{key.__qualname__}"


_container_lock = threading.Lock()
_container: Container | None = None


def get_container() -> Container:
    """Return the process-wide container, creating it on first use."""
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = Container()
    return _container


def reset_container() -> None:
    """Replace the process-wide container (intended for tests)."""
    global _container
    with _container_lock:
        if _container is not None:
            _container.clear()
        _container = Container()
