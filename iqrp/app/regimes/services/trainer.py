"""Training service for regime models."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

from iqrp.app.regimes.base.regime_model import RegimeModel
from iqrp.app.regimes.base.registry import ensure_regime_models_loaded, get_registry
from iqrp.app.regimes.config import RegimeSettings
from iqrp.app.regimes.services.serializer import RegimeSerializer


class RegimeTrainer:
    """Fit a registered regime model and optionally persist the artifact."""

    def __init__(self, settings: RegimeSettings | None = None) -> None:
        ensure_regime_models_loaded()
        self.settings = settings or RegimeSettings.default()
        self.serializer = RegimeSerializer()

    def train(
        self,
        frame: pl.DataFrame,
        *,
        model_name: str | None = None,
        feature_columns: list[str] | None = None,
        artifact_path: Path | None = None,
        **model_kwargs: object,
    ) -> RegimeModel:
        if not self.settings.enabled:
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "Regime framework disabled by configuration",
                code="REGIME_DISABLED",
            )
        name = model_name or self.settings.default_model
        model = get_registry().create(name, **model_kwargs)
        logger.info("regime_train_start model={}", name)
        model.fit(frame, feature_columns=feature_columns)
        if artifact_path is not None:
            self.serializer.save(model, artifact_path)
            logger.info("regime_train_saved path={}", artifact_path)
        return model
