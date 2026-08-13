"""Retry helpers for transient failures (sync and async)."""

from __future__ import annotations

import asyncio
import functools
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import ParamSpec, TypeVar

from loguru import logger

from iqrp.app.core.exceptions import IQRPError

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_EXCEPTIONS: tuple[type[BaseException], ...] = (IQRPError, TimeoutError, OSError)


def retry(
    *,
    attempts: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    jitter: float = 0.1,
    exceptions: Sequence[type[BaseException]] = DEFAULT_EXCEPTIONS,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry a synchronous callable with exponential backoff.

    Args:
        attempts: Maximum number of attempts (must be >= 1).
        delay: Initial delay in seconds between attempts.
        backoff: Multiplier applied to delay after each failure.
        jitter: Random fraction of delay added to reduce thundering herds.
        exceptions: Exception types that trigger a retry.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    exc_tuple = tuple(exceptions)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            wait = delay
            last_exc: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exc_tuple as exc:
                    last_exc = exc
                    if attempt == attempts:
                        break
                    sleep_for = wait + random.uniform(0, jitter * wait)  # noqa: S311
                    logger.warning(
                        "retry_attempt function={} attempt={} / {} error={} sleep_s={:.4f}",
                        func.__qualname__,
                        attempt,
                        attempts,
                        exc,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                    wait *= backoff
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


def async_retry(
    *,
    attempts: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
    jitter: float = 0.1,
    exceptions: Sequence[type[BaseException]] = DEFAULT_EXCEPTIONS,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Retry an async callable with exponential backoff."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    exc_tuple = tuple(exceptions)

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            wait = delay
            last_exc: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exc_tuple as exc:
                    last_exc = exc
                    if attempt == attempts:
                        break
                    sleep_for = wait + random.uniform(0, jitter * wait)  # noqa: S311
                    logger.warning(
                        "async_retry_attempt function={} attempt={} / {} error={} sleep_s={:.4f}",
                        func.__qualname__,
                        attempt,
                        attempts,
                        exc,
                        sleep_for,
                    )
                    await asyncio.sleep(sleep_for)
                    wait *= backoff
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
