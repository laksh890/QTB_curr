"""Live signal monitoring: decay, drift, performance, retirement, alerts."""

from __future__ import annotations

from iqrp.app.alpha.monitoring.alerts import build_alpha_alerts, summarize_alerts
from iqrp.app.alpha.monitoring.performance_decay import (
    max_drawdown,
    monitor_performance_decay,
    performance_decay_score,
    rolling_performance,
)
from iqrp.app.alpha.monitoring.retirement import batch_evaluate_retirement, evaluate_retirement
from iqrp.app.alpha.monitoring.signal_decay import (
    estimate_ic_half_life,
    ic_decay_curve,
    monitor_ic_decay,
    rolling_ic,
)
from iqrp.app.alpha.monitoring.signal_drift import (
    concept_drift_ic,
    monitor_signal_drift,
    position_drift,
    signal_distribution_drift,
)

__all__ = [
    "batch_evaluate_retirement",
    "build_alpha_alerts",
    "concept_drift_ic",
    "estimate_ic_half_life",
    "evaluate_retirement",
    "ic_decay_curve",
    "max_drawdown",
    "monitor_ic_decay",
    "monitor_performance_decay",
    "monitor_signal_drift",
    "performance_decay_score",
    "position_drift",
    "rolling_ic",
    "rolling_performance",
    "signal_distribution_drift",
    "summarize_alerts",
]
