"""Evaluate walk-forward folds and aggregate out-of-sample metrics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.backtesting.walk_forward.windows import WalkForwardWindow


FoldFn = Callable[[np.ndarray, np.ndarray], Mapping[str, Any]]


@dataclass
class FoldResult:
    fold_id: int
    metrics: dict[str, Any]
    window: WalkForwardWindow

    def to_dict(self) -> dict[str, Any]:
        out = self.window.to_dict()
        out["metrics"] = dict(self.metrics)
        return out


@dataclass
class WalkForwardReport:
    """Aggregated OOS walk-forward evaluation."""

    n_folds: int
    folds: list[FoldResult] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)
    mode: str = "rolling"
    purge: int = 0
    embargo: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_folds": int(self.n_folds),
            "mode": self.mode,
            "purge": int(self.purge),
            "embargo": int(self.embargo),
            "aggregate": dict(self.aggregate),
            "folds": [f.to_dict() for f in self.folds],
            "look_ahead_guard": "train_end <= prediction_timestamp (exclusive end)",
        }


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(float(x))


def aggregate_fold_metrics(fold_metrics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Mean / std / median across numeric metric keys shared by folds."""
    if not fold_metrics:
        return {}
    keys: set[str] = set()
    for m in fold_metrics:
        keys.update(k for k, v in m.items() if _is_number(v))
    out: dict[str, float] = {}
    for key in sorted(keys):
        vals = np.asarray(
            [float(m[key]) for m in fold_metrics if key in m and _is_number(m[key])],
            dtype=np.float64,
        )
        if vals.size == 0:
            continue
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out[f"{key}_median"] = float(np.median(vals))
        out[f"{key}_min"] = float(np.min(vals))
        out[f"{key}_max"] = float(np.max(vals))
    out["n_folds"] = float(len(fold_metrics))
    return out


class WalkForwardEvaluator:
    """Run a fold callable across windows and aggregate OOS metrics."""

    def evaluate(
        self,
        windows: Sequence[WalkForwardWindow],
        evaluate_fold: FoldFn,
        *,
        mode: str | None = None,
        purge: int | None = None,
        embargo: int | None = None,
    ) -> WalkForwardReport:
        folds: list[FoldResult] = []
        for w in windows:
            tr, te = w.as_index_pair()
            metrics = dict(evaluate_fold(tr, te))
            folds.append(FoldResult(fold_id=w.fold_id, metrics=metrics, window=w))

        agg = aggregate_fold_metrics([f.metrics for f in folds])
        mode_s = mode if mode is not None else (windows[0].mode if windows else "rolling")
        purge_v = int(purge if purge is not None else (windows[0].purge if windows else 0))
        embargo_v = int(
            embargo if embargo is not None else (windows[0].embargo if windows else 0)
        )
        return WalkForwardReport(
            n_folds=len(folds),
            folds=folds,
            aggregate=agg,
            mode=str(mode_s),
            purge=purge_v,
            embargo=embargo_v,
        )
