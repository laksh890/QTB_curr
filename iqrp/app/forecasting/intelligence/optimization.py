"""Optimization facade (aliases AutoML entrypoints)."""

from __future__ import annotations

from typing import Any

import polars as pl

from iqrp.app.forecasting.intelligence.automl import optimize_model
from iqrp.app.forecasting.intelligence.config import IntelligenceSettings


def run_optimization(
    name: str,
    frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    settings: IntelligenceSettings,
    search_space: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    return optimize_model(
        name,
        frame,
        feature_columns=feature_columns,
        target_column=target_column,
        settings=settings,
        search_space=search_space,
    )
