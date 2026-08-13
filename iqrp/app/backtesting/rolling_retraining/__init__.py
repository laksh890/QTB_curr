"""Rolling retraining: schedule triggers, versioned snapshots, OOS evaluation."""

from __future__ import annotations

from iqrp.app.backtesting.rolling_retraining.evaluator import (
    RetrainEpisode,
    RollingRetrainEvaluator,
    RollingRetrainReport,
    aggregate_episode_metrics,
)
from iqrp.app.backtesting.rolling_retraining.feature_snapshot import (
    FeatureSnapshot,
    FeatureSnapshotStore,
)
from iqrp.app.backtesting.rolling_retraining.model_registry import ModelRegistry, ModelSnapshot
from iqrp.app.backtesting.rolling_retraining.parameter_snapshot import (
    ParameterSnapshot,
    ParameterSnapshotStore,
)
from iqrp.app.backtesting.rolling_retraining.retrainer import RetrainEvent, RollingRetrainer
from iqrp.app.backtesting.rolling_retraining.schedule import (
    CompositeTrigger,
    DriftTrigger,
    PerformanceTrigger,
    RegimeTrigger,
    RetrainSchedule,
    RetrainTrigger,
    TimeTrigger,
    TriggerDecision,
)

__all__ = [
    "CompositeTrigger",
    "DriftTrigger",
    "FeatureSnapshot",
    "FeatureSnapshotStore",
    "ModelRegistry",
    "ModelSnapshot",
    "ParameterSnapshot",
    "ParameterSnapshotStore",
    "PerformanceTrigger",
    "RegimeTrigger",
    "RetrainEpisode",
    "RetrainEvent",
    "RetrainSchedule",
    "RetrainTrigger",
    "RollingRetrainEvaluator",
    "RollingRetrainReport",
    "RollingRetrainer",
    "TimeTrigger",
    "TriggerDecision",
    "aggregate_episode_metrics",
]
