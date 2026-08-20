"""Async token-bucket rate limiter for exchange REST calls."""

from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    """Token-bucket limiter shared across concurrent coroutines."""

    def __init__(self, rate_per_second: float, *, capacity: float | None = None) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        self._rate = rate_per_second
        self._capacity = capacity if capacity is not None else max(rate_per_second, 1.0)
        self._tokens = self._capacity
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until ``tokens`` are available, then consume them."""
        if tokens <= 0:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated_at
                self._updated_at = now
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait_for = deficit / self._rate
            await asyncio.sleep(wait_for)
