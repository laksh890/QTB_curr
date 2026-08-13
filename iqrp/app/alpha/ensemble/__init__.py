"""Alpha ensemble combination, weighting, correlation, clustering, redundancy."""

from __future__ import annotations

from iqrp.app.alpha.ensemble.clustering import (
    cluster_signals_from_series,
    correlation_distance,
    hierarchical_correlation_clusters,
    representative_per_cluster,
)
from iqrp.app.alpha.ensemble.correlation import (
    correlation_penalty_vector,
    drawdown_correlation_matrix,
    ic_correlation_matrix,
    position_correlation_matrix,
    prediction_correlation_matrix,
    return_correlation_matrix,
    signal_correlation_matrix,
)
from iqrp.app.alpha.ensemble.redundancy import (
    detect_nested_signals,
    feature_overlap,
    find_high_correlation_pairs,
    redundancy_report,
)
from iqrp.app.alpha.ensemble.signal_combination import (
    combine_from_metrics,
    combine_signals,
    majority_sign_combine,
    rank_average_combine,
)
from iqrp.app.alpha.ensemble.weighting import (
    DEFAULT_SCORE_WEIGHTS,
    compute_ensemble_weights,
    correlation_adjusted_weights,
    dynamic_weights,
    equal_weights,
    ic_weights,
    normalize_weights,
    regime_weights,
    risk_adjusted_weights,
    signal_quality_score,
)

__all__ = [
    "DEFAULT_SCORE_WEIGHTS",
    "cluster_signals_from_series",
    "combine_from_metrics",
    "combine_signals",
    "compute_ensemble_weights",
    "correlation_adjusted_weights",
    "correlation_distance",
    "correlation_penalty_vector",
    "detect_nested_signals",
    "drawdown_correlation_matrix",
    "dynamic_weights",
    "equal_weights",
    "feature_overlap",
    "find_high_correlation_pairs",
    "hierarchical_correlation_clusters",
    "ic_correlation_matrix",
    "ic_weights",
    "majority_sign_combine",
    "normalize_weights",
    "position_correlation_matrix",
    "prediction_correlation_matrix",
    "rank_average_combine",
    "redundancy_report",
    "regime_weights",
    "representative_per_cluster",
    "return_correlation_matrix",
    "risk_adjusted_weights",
    "signal_correlation_matrix",
    "signal_quality_score",
]
