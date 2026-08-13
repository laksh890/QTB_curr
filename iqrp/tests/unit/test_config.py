"""Unit tests for configuration models and loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from iqrp.app.config import AppSettings, Environment, load_config
from iqrp.app.config.loader import resolve_config_dir
from iqrp.app.core.exceptions import ConfigurationError


@pytest.mark.unit
def test_app_settings_defaults() -> None:
    settings = AppSettings()
    assert settings.environment is Environment.DEVELOPMENT
    assert settings.is_development
    assert not settings.is_production


@pytest.mark.unit
def test_load_development_config(config_dir: Path) -> None:
    settings = load_config(Environment.DEVELOPMENT, config_dir=config_dir)
    assert settings.environment is Environment.DEVELOPMENT
    assert settings.debug is True
    assert settings.logging.level == "DEBUG"


@pytest.mark.unit
def test_load_testing_config(config_dir: Path) -> None:
    settings = load_config(Environment.TESTING, config_dir=config_dir)
    assert settings.environment is Environment.TESTING
    assert settings.logging.file_logs is False
    assert settings.seed == 0


@pytest.mark.unit
def test_load_production_config(config_dir: Path) -> None:
    settings = load_config(Environment.PRODUCTION, config_dir=config_dir)
    assert settings.environment is Environment.PRODUCTION
    assert settings.debug is False
    assert settings.logging.json_logs is True


@pytest.mark.unit
def test_resolve_config_dir_missing() -> None:
    with pytest.raises(ConfigurationError):
        resolve_config_dir(Path("/nonexistent/iqrp-configs-xyz"))


@pytest.mark.unit
def test_overrides(config_dir: Path) -> None:
    settings = load_config(
        Environment.TESTING,
        config_dir=config_dir,
        overrides=["seed=99"],
    )
    assert settings.seed == 99
