"""Automatic model / horizon / feature / regime selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.intelligence.benchmark import (
    benchmark_candidates,
    benchmark_model,
)
from iqrp.app.forecasting.intelligence.config import IntelligenceSettings
from iqrp.app.forecasting.intelligence.ranking import RankedModel, rank_models
from iqrp.app.forecasting.intelligence.registry import list_discovered_models


@dataclass(slots=True)
class SelectionResult:
    best_model: str
    best_horizon: int
    best_features: list[str]
    best_regime_models: dict[str, str] = field(default_factory=dict)
    best_volatility_model: str | None = None
    best_ensemble: str | None = None
    ranked: list[RankedModel] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_model": self.best_model,
            "best_horizon": self.best_horizon,
            "best_features": list(self.best_features),
            "best_regime_models": dict(self.best_regime_models),
            "best_volatility_model": self.best_volatility_model,
            "best_ensemble": self.best_ensemble,
            "ranked": [r.to_dict() for r in self.ranked],
            "metadata": dict(self.metadata),
        }


def select_best(
    frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    settings: IntelligenceSettings,
    candidates: list[str] | None = None,
    horizons: list[int] | None = None,
) -> SelectionResult:
    results = benchmark_candidates(
        frame,
        feature_columns=feature_columns,
        target_column=target_column,
        settings=settings,
        candidates=candidates,
    )
    ranked = rank_models(
        [
            {"name": r.name, "family": r.family, "metrics": r.metrics, "metadata": r.metadata}
            for r in results
        ],
        settings.ranking,
    )
    best = ranked[0].name if ranked else (candidates[0] if candidates else "mock")
    # feature set: greedy drop-one if enough features
    best_features = _select_features(frame, feature_columns, target_column, best, settings)
    # horizon search
    best_horizon = _select_horizon(
        frame,
        best_features,
        target_column,
        best,
        settings,
        horizons or [settings.forecast.default_horizon],
    )
    # regime-specific
    regime_models = _select_regime_models(frame, best_features, target_column, settings, candidates)
    # volatility family preference
    vol_model = next((r.name for r in ranked if r.family == "volatility"), None)
    return SelectionResult(
        best_model=best,
        best_horizon=best_horizon,
        best_features=best_features,
        best_regime_models=regime_models,
        best_volatility_model=vol_model,
        best_ensemble=settings.ensemble.method if settings.ensemble.method != "none" else None,
        ranked=ranked,
        metadata={"n_candidates": len(results)},
    )


def _select_features(
    frame: pl.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    settings: IntelligenceSettings,
) -> list[str]:
    if len(feature_columns) <= 2:
        return list(feature_columns)
    base = benchmark_model(
        model_name,
        frame,
        feature_columns=feature_columns,
        target_column=target_column,
        settings=settings,
    )
    base_rmse = float(base.metrics.get("rmse", np.inf))
    kept = list(feature_columns)
    for col in feature_columns:
        trial = [c for c in kept if c != col]
        if len(trial) < 1:
            continue
        try:
            res = benchmark_model(
                model_name,
                frame,
                feature_columns=trial,
                target_column=target_column,
                settings=settings,
            )
            if float(res.metrics.get("rmse", np.inf)) <= base_rmse * 1.02:
                kept = trial
                base_rmse = float(res.metrics.get("rmse", base_rmse))
        except Exception:  # pragma: no cover
            continue
    return kept or list(feature_columns)


def _select_horizon(
    frame: pl.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    settings: IntelligenceSettings,
    horizons: list[int],
) -> int:
    # horizon quality proxied by prediction stability on last window
    best_h = max(int(horizons[0]), 1)
    best_score = float("inf")
    from iqrp.app.forecasting.intelligence.registry import create_model

    model = create_model(model_name)
    model.fit(frame, feature_columns=feature_columns, target_column=target_column)
    for h in horizons:
        h = max(int(h), 1)
        try:
            fc = model.forecast(frame, horizon=h)
            path = fc.path()
            score = float(np.std(path))
            # prefer moderate stability
            if score < best_score:
                best_score = score
                best_h = h
        except Exception:  # pragma: no cover
            continue
    return best_h


def _select_regime_models(
    frame: pl.DataFrame,
    feature_columns: list[str],
    target_column: str,
    settings: IntelligenceSettings,
    candidates: list[str] | None,
) -> dict[str, str]:
    col = settings.routing.regime_column
    if col not in frame.columns:
        return {}
    regimes = frame[col].unique().to_list()
    out: dict[str, str] = {}
    names = candidates or [m.name for m in list_discovered_models(settings)]
    # limit for speed
    names = names[: min(5, len(names))]
    for reg in regimes:
        sub = frame.filter(pl.col(col) == reg)
        if sub.height < 40:
            continue
        try:
            results = benchmark_candidates(
                sub,
                feature_columns=feature_columns,
                target_column=target_column,
                settings=settings,
                candidates=names,
            )
            ranked = rank_models(
                [{"name": r.name, "family": r.family, "metrics": r.metrics} for r in results],
                settings.ranking,
            )
            if ranked:
                out[str(reg)] = ranked[0].name
        except Exception:  # pragma: no cover
            continue
    return out
