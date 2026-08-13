"""In-process memoization decorators (sync and async).

These caches are process-local. Pass ``maxsize`` / ``ttl`` to bound growth.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import time
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")

_MISSING = object()


def _make_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    payload = repr((args, tuple(sorted(kwargs.items()))))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class _CacheEntry:
    value: Any
    expires_at: float | None


class _TTLCache:
    def __init__(self, maxsize: int | None, ttl: float | None) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Any:
        entry = self._store.get(key)
        if entry is None:
            return _MISSING
        if entry.expires_at is not None and time.monotonic() >= entry.expires_at:
            self._store.pop(key, None)
            return _MISSING
        return entry.value

    def set(self, key: str, value: Any) -> None:
        if (
            self._maxsize is not None
            and len(self._store) >= self._maxsize
            and key not in self._store
        ):
            oldest = next(iter(self._store))
            self._store.pop(oldest, None)
        expires = None if self._ttl is None else time.monotonic() + self._ttl
        self._store[key] = _CacheEntry(value=value, expires_at=expires)

    def clear(self) -> None:
        self._store.clear()


def cached(
    *,
    maxsize: int | None = 128,
    ttl: float | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Memoize a synchronous function."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        cache = _TTLCache(maxsize=maxsize, ttl=ttl)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            key = _make_key(args, kwargs)
            hit = cache.get(key)
            if hit is not _MISSING:
                return cast(R, hit)
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator


def async_cached(
    *,
    maxsize: int | None = 128,
    ttl: float | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Memoize an async function with in-flight request coalescing."""

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        cache = _TTLCache(maxsize=maxsize, ttl=ttl)
        inflight: dict[str, asyncio.Future[R]] = {}
        lock = asyncio.Lock()

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            key = _make_key(args, kwargs)
            hit = cache.get(key)
            if hit is not _MISSING:
                return cast(R, hit)

            async with lock:
                hit = cache.get(key)
                if hit is not _MISSING:
                    return cast(R, hit)
                existing = inflight.get(key)
                if existing is not None:
                    return await asyncio.shield(existing)
                loop = asyncio.get_running_loop()
                fut: asyncio.Future[R] = loop.create_future()
                inflight[key] = fut

            try:
                result = await func(*args, **kwargs)
            except Exception as exc:
                fut.set_exception(exc)
                async with lock:
                    inflight.pop(key, None)
                raise
            else:
                cache.set(key, result)
                fut.set_result(result)
                async with lock:
                    inflight.pop(key, None)
                return result

        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator


def hashable_key(*parts: Hashable) -> str:
    """Build a stable cache key from hashable parts."""
    return _make_key(tuple(parts), {})
