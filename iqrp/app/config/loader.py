"""Hydra-based configuration loader.

Composes ``configs/config.yaml`` with an environment overlay
(``development`` / ``testing`` / ``production``) and validates the result
into an immutable :class:`AppSettings` instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from iqrp.app.config.settings import AppSettings, Environment
from iqrp.app.core.exceptions import ConfigurationError


def resolve_config_dir(explicit: Path | None = None) -> Path:
    """Locate the Hydra config directory.

    Resolution order:
    1. Explicit path argument
    2. ``IQRP_CONFIG_DIR`` environment variable
    3. Package-relative ``iqrp/configs``
    """
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_dir():
            raise ConfigurationError(
                f"Config directory does not exist: {path}",
                code="CONFIG_DIR_MISSING",
                details={"path": str(path)},
            )
        return path

    import os

    env_dir = os.environ.get("IQRP_CONFIG_DIR")
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
        if not path.is_dir():
            raise ConfigurationError(
                f"IQRP_CONFIG_DIR does not exist: {path}",
                code="CONFIG_DIR_MISSING",
                details={"path": str(path)},
            )
        return path

    package_root = Path(__file__).resolve().parents[2]
    candidate = package_root / "configs"
    if candidate.is_dir():
        return candidate

    raise ConfigurationError(
        "Unable to locate IQRP config directory",
        code="CONFIG_DIR_MISSING",
        details={"searched": str(candidate)},
    )


def load_config(
    environment: Environment | str = Environment.DEVELOPMENT,
    *,
    config_dir: Path | None = None,
    overrides: list[str] | None = None,
) -> AppSettings:
    """Compose and validate application settings.

    Args:
        environment: Target environment name or enum value.
        config_dir: Optional override for the Hydra config directory.
        overrides: Additional Hydra override strings (e.g. ``debug=true``).

    Returns:
        Validated, frozen :class:`AppSettings`.
    """
    env = Environment(str(environment).lower())
    cfg_dir = resolve_config_dir(config_dir)
    hydra_overrides = [f"environment={env.value}"]
    if overrides:
        hydra_overrides.extend(overrides)

    # Hydra forbids nested initialize without clearing prior state.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    try:
        with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.3"):
            cfg = compose(config_name="config", overrides=hydra_overrides)
    except Exception as exc:
        raise ConfigurationError(
            f"Failed to compose Hydra configuration: {exc}",
            code="CONFIG_COMPOSE_FAILED",
            details={"environment": env.value, "config_dir": str(cfg_dir)},
        ) from exc
    finally:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

    raw: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    if not isinstance(raw, dict):
        raise ConfigurationError(
            "Hydra composition did not yield a mapping",
            code="CONFIG_INVALID_SHAPE",
        )

    # Environment overlay files set top-level keys; ensure environment field.
    raw["environment"] = env.value

    try:
        return AppSettings.model_validate(raw)
    except Exception as exc:
        raise ConfigurationError(
            f"Configuration validation failed: {exc}",
            code="CONFIG_VALIDATION_FAILED",
            details={"environment": env.value},
        ) from exc
