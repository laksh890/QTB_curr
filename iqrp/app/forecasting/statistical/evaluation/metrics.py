"""Evaluation helpers re-exporting framework metrics for statistical models."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.forecasting.base.evaluator import (
    ForecastEvaluator,
    directional_accuracy,
    mae,
    mape,
    max_drawdown,
    profit_factor,
    r2_score,
    rmse,
    sharpe_ratio,
    smape,
)


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    ev = ForecastEvaluator()
    report = ev.evaluate(y_true, y_pred, task="regression", probabilities=probabilities)
    metrics = dict(report.metrics)
    # ensure financial keys present
    metrics.setdefault("directional_accuracy", directional_accuracy(y_true, y_pred))
    metrics.setdefault("sharpe", sharpe_ratio(y_true, y_pred))
    metrics.setdefault("profit_factor", profit_factor(y_true, y_pred))
    metrics.setdefault("max_drawdown", max_drawdown(y_true, y_pred))
    return metrics


def summary_table(
    results: dict[str, dict[str, float]], *, primary: str = "rmse"
) -> list[dict[str, Any]]:
    return ForecastEvaluator().benchmark(results, primary=primary)


__all__ = [
    "directional_accuracy",
    "evaluate_forecast",
    "mae",
    "mape",
    "max_drawdown",
    "profit_factor",
    "r2_score",
    "rmse",
    "sharpe_ratio",
    "smape",
    "summary_table",
]
