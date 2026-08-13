"""Unit tests for common utilities."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from iqrp.app.common.cache import async_cached, cached
from iqrp.app.common.datetime_utils import ensure_utc, parse_datetime, to_iso8601, utc_now
from iqrp.app.common.retry import async_retry, retry
from iqrp.app.common.singleton import SingletonMeta, singleton
from iqrp.app.common.timer import Timer, timed
from iqrp.app.common.uuid_utils import new_id, new_uuid
from iqrp.app.core.exceptions import DataError, ValidationError


@pytest.mark.unit
def test_singleton_decorator() -> None:
    @singleton
    class Registry:
        def __init__(self) -> None:
            self.n = 0

    a = Registry()
    b = Registry()
    assert a is b
    Registry.reset_singleton()  # type: ignore[attr-defined]
    c = Registry()
    assert c is not a


@pytest.mark.unit
def test_singleton_metaclass() -> None:
    class Hub(metaclass=SingletonMeta):
        pass

    assert Hub() is Hub()
    Hub.reset_instance()
    # After reset, a new instance is created
    first = Hub()
    Hub.reset_instance()
    second = Hub()
    assert first is not second


@pytest.mark.unit
def test_timer_context() -> None:
    with Timer("t") as timer:
        assert timer.elapsed >= 0.0
    assert timer.elapsed >= 0.0


@pytest.mark.unit
def test_timed_context_manager() -> None:
    with timed("block") as timer:
        assert timer.name == "block"
    assert timer.elapsed >= 0.0


@pytest.mark.unit
def test_retry_eventually_succeeds() -> None:
    state = {"n": 0}

    @retry(attempts=3, delay=0.01, backoff=1.0, jitter=0.0, exceptions=(DataError,))
    def flaky() -> str:
        state["n"] += 1
        if state["n"] < 3:
            raise DataError("transient")
        return "ok"

    assert flaky() == "ok"
    assert state["n"] == 3


@pytest.mark.unit
def test_retry_exhausts() -> None:
    @retry(attempts=2, delay=0.01, backoff=1.0, jitter=0.0, exceptions=(DataError,))
    def always_fail() -> None:
        raise DataError("nope")

    with pytest.raises(DataError):
        always_fail()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_retry() -> None:
    state = {"n": 0}

    @async_retry(attempts=3, delay=0.01, backoff=1.0, jitter=0.0, exceptions=(DataError,))
    async def flaky() -> int:
        state["n"] += 1
        if state["n"] < 2:
            raise DataError("transient")
        return 1

    assert await flaky() == 1


@pytest.mark.unit
def test_cached_memoizes() -> None:
    calls = {"n": 0}

    @cached(maxsize=8)
    def compute(x: int) -> int:
        calls["n"] += 1
        return x * 2

    assert compute(2) == 4
    assert compute(2) == 4
    assert calls["n"] == 1
    compute.cache_clear()  # type: ignore[attr-defined]
    assert compute(2) == 4
    assert calls["n"] == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_cached() -> None:
    calls = {"n": 0}

    @async_cached(maxsize=8)
    async def compute(x: int) -> int:
        calls["n"] += 1
        await asyncio.sleep(0)
        return x + 1

    assert await compute(1) == 2
    assert await compute(1) == 2
    assert calls["n"] == 1


@pytest.mark.unit
def test_uuid_helpers() -> None:
    u = new_uuid(4)
    assert u.version == 4
    assert new_id("run").startswith("run_")
    u7 = new_uuid(7)
    assert u7.version == 7


@pytest.mark.unit
def test_datetime_helpers() -> None:
    now = utc_now()
    assert now.tzinfo is not None
    naive = datetime(2024, 1, 1, 12, 0, 0)
    assert ensure_utc(naive).tzinfo == UTC
    assert to_iso8601(ensure_utc(naive)).endswith("Z")
    parsed = parse_datetime("2024-01-01T12:00:00Z")
    assert parsed == datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    with pytest.raises(ValidationError):
        parse_datetime("not-a-date")
