"""Causal walk-forward backtesting: rolling / expanding / anchored / purged folds."""

from __future__ import annotations

from iqrp.app.backtesting.walk_forward.embargo import (
    apply_embargo,
    embargo_after_test,
    embargo_range,
    embargo_splits,
)
from iqrp.app.backtesting.walk_forward.engine import WalkForwardEngine
from iqrp.app.backtesting.walk_forward.evaluator import (
    FoldResult,
    WalkForwardEvaluator,
    WalkForwardReport,
    aggregate_fold_metrics,
)
from iqrp.app.backtesting.walk_forward.purge import (
    apply_purge,
    purge_range,
    purge_train_indices,
    purged_kfold_splits,
)
from iqrp.app.backtesting.walk_forward.test_window import TestWindow
from iqrp.app.backtesting.walk_forward.training_window import TrainingWindow
from iqrp.app.backtesting.walk_forward.validation_window import ValidationWindow
from iqrp.app.backtesting.walk_forward.windows import (
    WalkForwardWindow,
    assert_no_future_training,
    generate_windows,
)

__all__ = [
    "FoldResult",
    "TestWindow",
    "TrainingWindow",
    "ValidationWindow",
    "WalkForwardEngine",
    "WalkForwardEvaluator",
    "WalkForwardReport",
    "WalkForwardWindow",
    "aggregate_fold_metrics",
    "apply_embargo",
    "apply_purge",
    "assert_no_future_training",
    "embargo_after_test",
    "embargo_range",
    "embargo_splits",
    "generate_windows",
    "purge_range",
    "purge_train_indices",
    "purged_kfold_splits",
]
