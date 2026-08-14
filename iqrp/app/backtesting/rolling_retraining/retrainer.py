"""Rolling model retraining with schedule triggers and versioned snapshots.

Look-ahead prevention
---------------------
A model registered at ``trained_through=t`` may only be evaluated on indices
``> t``. Retrains never use future observations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.backtesting.rolling_retraining.evaluator import (
    RetrainEpisode,
    RollingRetrainEvaluator,
    RollingRetrainReport,
)
from iqrp.app.backtesting.rolling_retraining.feature_snapshot import FeatureSnapshotStore
from iqrp.app.backtesting.rolling_retraining.model_registry import ModelRegistry, ModelSnapshot
from iqrp.app.backtesting.rolling_retraining.parameter_snapshot import ParameterSnapshotStore
from iqrp.app.backtesting.rolling_retraining.schedule import (
    RetrainSchedule,
    TriggerDecision,
)

TrainFn = Callable[[np.ndarray, np.ndarray | None, dict[str, Any]], Any]
"""``train_fn(X_train, y_train, params) -> model``"""

PredictFn = Callable[[Any, np.ndarray], Any]
"""``predict_fn(model, X_test) -> predictions``"""

ScoreFn = Callable[[Any, np.ndarray, np.ndarray | None], Mapping[str, float]]
"""``score_fn(model, X_test, y_test) -> metrics``"""


@dataclass
class RetrainEvent:
    t: int
    decision: TriggerDecision
    snapshot: ModelSnapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": int(self.t),
            "decision": self.decision.to_dict(),
            "snapshot": self.snapshot.to_dict(),
        }


@dataclass
class RollingRetrainer:
    """Orchestrate schedule-driven retraining with an in-memory model registry.

    Parameters
    ----------
    schedule:
        Retrain schedule (time / performance / drift / regime / composite).
    train_window:
        Fixed lookback for rolling training. If ``None``, uses expanding
        history from ``origin`` to ``t`` (exclusive of prediction bar).
    origin:
        First index eligible for training history.
    """

    schedule: RetrainSchedule = field(default_factory=lambda: RetrainSchedule(every=20))
    registry: ModelRegistry = field(default_factory=ModelRegistry)
    features: FeatureSnapshotStore = field(default_factory=FeatureSnapshotStore)
    parameters: ParameterSnapshotStore = field(default_factory=ParameterSnapshotStore)
    evaluator: RollingRetrainEvaluator = field(default_factory=RollingRetrainEvaluator)
    train_window: int | None = None
    origin: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    events: list[RetrainEvent] = field(default_factory=list)
    episodes: list[RetrainEpisode] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.train_window is not None and int(self.train_window) < 1:
            raise ValueError("train_window must be >= 1 when provided")
        self.origin = max(int(self.origin), 0)

    def training_slice(self, t: int) -> tuple[int, int]:
        """Return half-open ``[start, end)`` train bounds ending at ``t`` (exclusive).

        ``end == t`` guarantees training never includes the prediction timestamp.
        """
        t = int(t)
        if t <= self.origin:
            raise ValueError(f"t={t} must be > origin={self.origin}")
        end = t  # exclusive — NO FUTURE TRAINING
        if self.train_window is None:
            start = self.origin
        else:
            start = max(self.origin, end - int(self.train_window))
        if end <= start:
            raise ValueError(f"Empty training window at t={t}")
        return start, end

    def maybe_retrain(
        self,
        t: int,
        *,
        X: Any,
        y: Any | None = None,
        train_fn: TrainFn,
        context: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        force: bool = False,
    ) -> ModelSnapshot | None:
        """Evaluate schedule at time ``t`` and retrain if triggered.

        Training uses indices ``[start, t)`` only.
        """
        t = int(t)
        ctx = dict(context or {})
        active = self.registry.active()
        last_t = int(active.trained_through) if active is not None else self.origin - 1
        ctx.setdefault("t", t)
        ctx.setdefault("index", t)
        ctx.setdefault("last_retrain_t", last_t)
        ctx.setdefault("trained_through", last_t)

        decision = (
            TriggerDecision(True, "manual", "force", {})
            if force
            else self.schedule.should_retrain(ctx)
        )
        if not decision.should_retrain:
            return None

        start, end = self.training_slice(t)
        X_arr = np.asarray(X)
        y_arr = None if y is None else np.asarray(y)
        X_tr = X_arr[start:end]
        y_tr = None if y_arr is None else y_arr[start:end]

        param_snap = self.parameters.save(self.params, source=decision.kind)
        feat_snap = self.features.save(
            X_tr, start=start, end=end, columns=columns, metadata={"trigger": decision.kind}
        )
        model = train_fn(X_tr, y_tr, dict(self.params))
        # trained_through is the last inclusive train index (= end - 1)
        trained_through = end - 1
        if trained_through >= t:
            raise ValueError(
                f"NO FUTURE TRAINING: trained_through={trained_through} must be < t={t}"
            )
        snap = self.registry.register(
            model,
            trained_through=trained_through,
            metrics=dict(ctx.get("metrics") or {}),
            metadata={"train_start": start, "train_end": end},
            feature_version=feat_snap.version,
            parameter_version=param_snap.version,
            trigger=str(decision.kind),
            activate=True,
        )
        self.schedule.record_retrain()
        self.events.append(RetrainEvent(t=t, decision=decision, snapshot=snap))
        return snap

    def predict_at(
        self,
        t: int,
        *,
        X: Any,
        predict_fn: PredictFn,
        version: int | None = None,
    ) -> Any:
        """Predict at index ``t`` using a model trained strictly before ``t``."""
        snap = self.registry.get(version)
        if snap is None:
            raise RuntimeError("No active model in registry")
        if int(snap.trained_through) >= int(t):
            raise ValueError(
                f"NO FUTURE TRAINING: model trained_through={snap.trained_through} "
                f">= prediction t={t}"
            )
        X_arr = np.asarray(X)
        return predict_fn(snap.model, X_arr[t : t + 1])

    def run(
        self,
        *,
        X: Any,
        y: Any | None = None,
        train_fn: TrainFn,
        score_fn: ScoreFn | None = None,
        predict_fn: PredictFn | None = None,
        start: int | None = None,
        end: int | None = None,
        warm_start_train: bool = True,
        context_fn: Callable[[int, ModelSnapshot | None], dict[str, Any]] | None = None,
        columns: list[str] | None = None,
        as_dict: bool = True,
    ) -> dict[str, Any] | RollingRetrainReport:
        """Walk forward through time, retraining on schedule and scoring OOS segments.

        At each time ``t``, the active model (trained through ``< t``) may be
        scored on bar ``t``. Retrain decisions also occur at ``t`` using data
        strictly before ``t``.
        """
        X_arr = np.asarray(X)
        n = int(X_arr.shape[0])
        y_arr = None if y is None else np.asarray(y)
        t0 = int(start) if start is not None else max(self.origin + 1, (self.train_window or 1))
        t1 = int(end) if end is not None else n
        t0 = max(t0, self.origin + 1)
        t1 = min(t1, n)

        if warm_start_train and self.registry.active() is None and t0 > self.origin:
            self.maybe_retrain(
                t0,
                X=X_arr,
                y=y_arr,
                train_fn=train_fn,
                context={"t": t0, "last_retrain_t": self.origin - 1},
                columns=columns,
                force=True,
            )

        episode_start: int | None = None
        current_version: int | None = (
            None if self.registry.active() is None else int(self.registry.active().version)
        )
        if current_version is not None:
            episode_start = t0

        oos_metrics_acc: list[dict[str, float]] = []

        for t in range(t0, t1):
            active = self.registry.active()
            ctx = {"t": t, "last_retrain_t": active.trained_through if active else self.origin - 1}
            if context_fn is not None:
                ctx.update(context_fn(t, active))

            # Score current bar with the model trained strictly before t.
            if active is not None and score_fn is not None and int(active.trained_through) < t:
                X_te = X_arr[t : t + 1]
                y_te = None if y_arr is None else y_arr[t : t + 1]
                m = dict(score_fn(active.model, X_te, y_te))
                oos_metrics_acc.append({k: float(v) for k, v in m.items() if _numeric(v)})

            snap = self.maybe_retrain(
                t,
                X=X_arr,
                y=y_arr,
                train_fn=train_fn,
                context=ctx,
                columns=columns,
                force=False,
            )
            if snap is not None:
                # Close previous episode at t (exclusive eval end).
                if current_version is not None and episode_start is not None and episode_start < t:
                    prev = self.registry.get(current_version)
                    self.episodes.append(
                        RetrainEpisode(
                            version=current_version,
                            trained_through=int(prev.trained_through) if prev else -1,
                            eval_start=int(episode_start),
                            eval_end=int(t),
                            trigger=prev.trigger if prev else None,
                            metrics=_mean_metrics(oos_metrics_acc),
                        )
                    )
                    oos_metrics_acc = []
                current_version = int(snap.version)
                episode_start = t  # predictions from this model start at next bars; allow t+?
                # Model trained through t-1 can predict at t immediately after retrain.
                # Keep episode_start = t so eval uses indices >= t with trained_through = t-1.
                episode_start = t

        # Close final episode.
        if current_version is not None and episode_start is not None and episode_start < t1:
            prev = self.registry.get(current_version)
            self.episodes.append(
                RetrainEpisode(
                    version=current_version,
                    trained_through=int(prev.trained_through) if prev else -1,
                    eval_start=int(episode_start),
                    eval_end=int(t1),
                    trigger=prev.trigger if prev else None,
                    metrics=_mean_metrics(oos_metrics_acc),
                )
            )

        report = self.evaluator.evaluate(self.episodes)
        if as_dict:
            out = report.to_dict()
            out["events"] = [e.to_dict() for e in self.events]
            out["n_models"] = self.registry.size
            out["active_version"] = self.registry.active_version
            return out
        return report

    def active_model(self) -> Any | None:
        snap = self.registry.active()
        return None if snap is None else snap.model


def _numeric(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v))


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = set()
    for r in rows:
        keys.update(r.keys())
    out: dict[str, float] = {}
    for k in sorted(keys):
        vals = [float(r[k]) for r in rows if k in r]
        if vals:
            out[k] = float(np.mean(vals))
    return out
