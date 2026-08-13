"""Integration tests for configuration composition across environments."""

from __future__ import annotations

from pathlib import Path

import pytest

from iqrp.app.config import Environment, load_config
from iqrp.app.core.container import get_container
from iqrp.app.logging import setup_logging


@pytest.mark.integration
@pytest.mark.parametrize("env", list(Environment))
def test_all_environments_load(config_dir: Path, env: Environment) -> None:
    settings = load_config(env, config_dir=config_dir)
    assert settings.environment is env
    assert settings.name == "iqrp"
    setup_logging(settings.logging)


@pytest.mark.integration
def test_container_registers_settings(settings: object) -> None:
    container = get_container()
    container.register_instance("settings", settings)
    assert container.resolve("settings") is settings
