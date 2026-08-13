"""Multi-metric ranking for forecast candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.forecasting.base import evaluator as ev
from iqrp.app.forecasting.intelligence.config import RankingConfig


@dataclass(slots=True)
class RankedModel:
    name: str
    metrics: dict[str, float]
    score: float
    rank: int = 0
    family: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metrics": dict(self.metrics),
            "score": self.score,
            "rank": self.rank,
            "family": self.family,
            "metadata": dict(self.metadata),
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    probabilities: np.ndarray | None = None,
    latency_ms: float = 0.0,
    memory_mb: float = 0.0,
    inference_cost: float = 0.0,
) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(yt.size, yp.size)
    yt, yp = yt[:n], yp[:n]
    out: dict[str, float] = {
        "mae": ev.mae(yt, yp),
        "mse": ev.mse(yt, yp),
        "rmse": ev.rmse(yt, yp),
        "mape": ev.mape(yt, yp),
        "smape": ev.smape(yt, yp),
        "r2": ev.r2_score(yt, yp),
        "directional_accuracy": ev.directional_accuracy(yt, yp),
        "sharpe": ev.sharpe_ratio(yt, yp),
        "sortino": ev.sortino_ratio(yt, yp),
        "profit_factor": ev.profit_factor(yt, yp),
        "max_drawdown": ev.max_drawdown(yt, yp),
        "latency_ms": float(latency_ms),
        "memory_mb": float(memory_mb),
        "inference_cost": float(inference_cost),
        "prediction_stability": float(1.0 / (1.0 + np.std(yp))) if yp.size else 0.0,
        "n": float(n),
    }
    if probabilities is not None:
        P = np.asarray(probabilities, dtype=np.float64)
        scores = P[:, -1] if P.ndim == 2 and P.shape[1] >= 2 else P.reshape(-1)
        labels = (yt > np.median(yt)).astype(np.float64)[: scores.size]
        out["brier"] = ev.brier_score(scores[: labels.size], labels)
        out["log_loss"] = ev.log_loss(scores[: labels.size], labels)
        out["calibration_error"] = ev.expected_calibration_error(scores[: labels.size], labels)
    return out


def composite_score(metrics: dict[str, float], config: RankingConfig) -> float:
    """Lower score is better after flipping higher-is-better metrics."""
    total = 0.0
    weight_sum = 0.0
    for name, w in config.weights.items():
        if name not in metrics or not np.isfinite(metrics[name]):
            continue
        val = float(metrics[name])
        if name in config.higher_is_better:
            val = -val
        total += w * val
        weight_sum += abs(w)
    if weight_sum <= 0:
        primary = config.primary
        val = float(metrics.get(primary, np.inf))
        if primary in config.higher_is_better:
            val = -val
        return val
    return total / weight_sum


def rank_models(
    results: list[dict[str, Any]],
    config: RankingConfig | None = None,
) -> list[RankedModel]:
    cfg = config or RankingConfig()
    ranked: list[RankedModel] = []
    for r in results:
        metrics = dict(r.get("metrics") or {})
        scored = RankedModel(
            name=str(r.get("name", "")),
            metrics=metrics,
            score=composite_score(metrics, cfg),
            family=str(r.get("family", "")),
            metadata=dict(r.get("metadata") or {}),
        )
        ranked.append(scored)
    ranked.sort(key=lambda x: (x.score, x.name))
    for i, item in enumerate(ranked, start=1):
        item.rank = i
    return ranked


def leaderboard_table(ranked: list[RankedModel]) -> list[dict[str, Any]]:
    return [r.to_dict() for r in ranked]
