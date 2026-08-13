"""High-resolution timing utilities for research and diagnostics."""

from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import ParamSpec, TypeVar

from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(slots=True)
class Timer:
    """Wall-clock timer usable as a context manager or explicit stopwatch."""

    name: str = "timer"
    _start: float | None = field(default=None, init=False, repr=False)
    _elapsed: float | None = field(default=None, init=False, repr=False)

    def start(self) -> Timer:
        self._start = time.perf_counter()
        self._elapsed = None
        return self

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer has not been started")
        self._elapsed = time.perf_counter() - self._start
        return self._elapsed

    @property
    def elapsed(self) -> float:
        if self._elapsed is not None:
            return self._elapsed
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start

    def __enter__(self) -> Timer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        seconds = self.stop()
        logger.debug("timer_complete name={} elapsed_s={:.6f}", self.name, seconds)


@contextmanager
def timed(name: str = "block") -> Iterator[Timer]:
    """Context manager that logs elapsed wall time on exit."""
    timer = Timer(name=name)
    timer.start()
    try:
        yield timer
    finally:
        seconds = timer.stop()
        logger.debug("timed_block name={} elapsed_s={:.6f}", name, seconds)


def timed_sync(name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that logs execution time of a synchronous function."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        label = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                logger.debug("timed_call name={} elapsed_s={:.6f}", label, elapsed)

        return wrapper

    return decorator


def timed_async(
    name: str | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorator that logs execution time of an async function."""

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        label = name or func.__qualname__

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            started = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                logger.debug("timed_async_call name={} elapsed_s={:.6f}", label, elapsed)

        return wrapper

    return decorator


# Public alias matching the requested "Timer" / timed API surface.
timed_callable = timed_sync
