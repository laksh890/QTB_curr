"""Unit tests for the DI container."""

from __future__ import annotations

import pytest

from iqrp.app.core.container import Container, get_container, reset_container
from iqrp.app.core.exceptions import ConfigurationError


class _Service:
    def __init__(self, value: int = 1) -> None:
        self.value = value


@pytest.mark.unit
def test_register_and_resolve_singleton() -> None:
    container = Container()
    counter = {"n": 0}

    def provider() -> _Service:
        counter["n"] += 1
        return _Service(counter["n"])

    container.register(_Service, provider, singleton=True)
    a = container.resolve(_Service)
    b = container.resolve(_Service)
    assert a is b
    assert counter["n"] == 1


@pytest.mark.unit
def test_factory_scope_creates_new_instances() -> None:
    container = Container()
    container.register(_Service, lambda: _Service(7), singleton=False)
    a = container.resolve(_Service)
    b = container.resolve(_Service)
    assert a is not b
    assert a.value == 7


@pytest.mark.unit
def test_register_instance_and_has() -> None:
    container = Container()
    svc = _Service(42)
    container.register_instance("svc", svc)
    assert container.has("svc")
    assert container.resolve("svc") is svc


@pytest.mark.unit
def test_missing_provider_raises() -> None:
    container = Container()
    with pytest.raises(ConfigurationError) as exc_info:
        container.resolve("missing")
    assert exc_info.value.code == "DI_PROVIDER_MISSING"


@pytest.mark.unit
def test_process_container_reset() -> None:
    c1 = get_container()
    c1.register_instance("x", 1)
    reset_container()
    c2 = get_container()
    assert c1 is not c2
    assert not c2.has("x")
