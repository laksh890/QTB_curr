"""Shared pytest fixtures for IQRP."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from iqrp.app.config import AppSettings, Environment, load_config
from iqrp.app.core.container import reset_container
from iqrp.app.logging.setup import reset_logging_state, setup_logging


@pytest.fixture(autouse=True)
def _isolate_container() -> Iterator[None]:
    """Ensure each test starts with a clean DI container."""
    reset_container()
    yield
    reset_container()


@pytest.fixture(autouse=True)
def _isolate_logging() -> Iterator[None]:
    """Reset logging between tests to avoid sink leakage."""
    reset_logging_state()
    yield
    reset_logging_state()


@pytest.fixture
def config_dir() -> Path:
    """Return the package Hydra config directory."""
    return Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def settings(config_dir: Path) -> AppSettings:
    """Load testing-environment settings."""
    return load_config(Environment.TESTING, config_dir=config_dir)


@pytest.fixture
def configured_logging(settings: AppSettings) -> None:
    """Configure logging from testing settings."""
    setup_logging(settings.logging)
