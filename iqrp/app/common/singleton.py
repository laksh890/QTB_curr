"""Singleton helpers without ambient global mutable state.

Prefer dependency injection via :class:`~iqrp.app.core.container.Container`
for application services. These helpers exist for rare, intentional
single-instance types (e.g. process-local registries).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, ClassVar, cast, overload


class SingletonMeta(type):
    """Metaclass that guarantees one instance per class."""

    _instances: ClassVar[dict[type[Any], Any]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def reset_instance(cls) -> None:
        """Drop the cached instance (test support)."""
        with cls._lock:
            cls._instances.pop(cls, None)


@overload
def singleton[T](cls: type[T], /) -> type[T]: ...


@overload
def singleton[T](*, resettable: bool = True) -> Callable[[type[T]], type[T]]: ...


def singleton[T](
    cls: type[T] | None = None,
    /,
    *,
    resettable: bool = True,
) -> type[T] | Callable[[type[T]], type[T]]:
    """Class decorator that enforces singleton construction.

    Example::

        @singleton
        class Registry:
            ...
    """

    def decorator(target: type[T]) -> type[T]:
        lock = threading.Lock()
        holder: dict[str, T | None] = {"instance": None}

        original_new = target.__new__

        def __new__(cls_: type[T], *args: Any, **kwargs: Any) -> T:  # noqa: N807
            if holder["instance"] is None:
                with lock:
                    if holder["instance"] is None:
                        if original_new is object.__new__:
                            instance = original_new(cls_)
                        else:
                            instance = original_new(cls_, *args, **kwargs)
                        holder["instance"] = instance
            return cast(T, holder["instance"])

        target.__new__ = staticmethod(__new__)  # type: ignore[assignment]

        if resettable:

            @classmethod  # type: ignore[misc]
            def reset_singleton(cls_: type[T]) -> None:
                with lock:
                    holder["instance"] = None

            target.reset_singleton = reset_singleton  # type: ignore[attr-defined]

        return target

    if cls is not None:
        return decorator(cls)
    return decorator
