"""Evaluate rolling-retrain episodes and aggregate live OOS metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(float(x))


@dataclass
class RetrainEpisode:
    """One retrain event and the OOS segment that followed it."""

    version: int
    trained_through: int
    eval_start: int
    eval_end: int
    trigger: str | None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "trained_through": int(self.trained_through),
            "eval_start": int(self.eval_start),
            "eval_end": int(self.eval_end),
            "trigger": self.trigger,
            "metrics": dict(self.metrics),
        }


@dataclass
class RollingRetrainReport:
    n_episodes: int
    episodes: list[RetrainEpisode] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_episodes": int(self.n_episodes),
            "aggregate": dict(self.aggregate),
            "episodes": [e.to_dict() for e in self.episodes],
            "look_ahead_guard": "trained_through < eval_start",
        }


def aggregate_episode_metrics(
    episodes: Sequence[Mapping[str, Any] | RetrainEpisode],
) -> dict[str, float]:
    metrics_list: list[Mapping[str, Any]] = []
    for e in episodes:
        if isinstance(e, RetrainEpisode):
            metrics_list.append(e.metrics)
        else:
            metrics_list.append(e.get("metrics", e))  # type: ignore[arg-type]
    if not metrics_list:
        return {}
    keys: set[str] = set()
    for m in metrics_list:
        keys.update(k for k, v in m.items() if _is_number(v))
    out: dict[str, float] = {}
    for key in sorted(keys):
        vals = np.asarray(
            [float(m[key]) for m in metrics_list if key in m and _is_number(m[key])],
            dtype=np.float64,
        )
        if vals.size == 0:
            continue
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out[f"{key}_median"] = float(np.median(vals))
    out["n_episodes"] = float(len(metrics_list))
    return out


class RollingRetrainEvaluator:
    """Aggregate metrics across retrain episodes."""

    def evaluate(self, episodes: Sequence[RetrainEpisode]) -> RollingRetrainReport:
        for ep in episodes:
            if int(ep.trained_through) >= int(ep.eval_start):
                raise ValueError(
                    f"NO FUTURE TRAINING: trained_through={ep.trained_through} "
                    f">= eval_start={ep.eval_start}"
                )
        agg = aggregate_episode_metrics(episodes)
        return RollingRetrainReport(
            n_episodes=len(episodes),
            episodes=list(episodes),
            aggregate=agg,
        )
