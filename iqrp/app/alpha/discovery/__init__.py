"""Alpha candidate discovery: templates and orchestration."""

from iqrp.app.alpha.discovery.alternative import (
    alternative_change_signal,
    alternative_zscore_signal,
    apply_publication_lag,
    sentiment_pressure_signal,
)
from iqrp.app.alpha.discovery.candidate_generator import (
    CandidateGenerator,
    DiscoveryResult,
    generate_candidates,
)
from iqrp.app.alpha.discovery.cross_sectional import (
    cross_sectional_rank_signal,
    cross_sectional_zscore_signal,
    long_short_spread,
)
from iqrp.app.alpha.discovery.event_based import (
    earnings_drift_proxy,
    event_impulse_signal,
    surprise_signal,
)
from iqrp.app.alpha.discovery.statistical import (
    StatisticalCandidate,
    candidates_to_signals,
    screen_features,
)
from iqrp.app.alpha.discovery.symbolic import (
    delay,
    diff,
    evaluate_expression,
    lag,
    rank,
    ratio,
    rolling_mean,
    rolling_std,
    rolling_sum,
    zscore,
)
from iqrp.app.alpha.discovery.time_series import (
    build_time_series_candidates,
    mean_reversion_signal,
    momentum_signal,
    trend_signal,
    volatility_signal,
    volume_signal,
)

__all__ = [
    "CandidateGenerator",
    "DiscoveryResult",
    "StatisticalCandidate",
    "alternative_change_signal",
    "alternative_zscore_signal",
    "apply_publication_lag",
    "build_time_series_candidates",
    "candidates_to_signals",
    "cross_sectional_rank_signal",
    "cross_sectional_zscore_signal",
    "delay",
    "diff",
    "earnings_drift_proxy",
    "evaluate_expression",
    "event_impulse_signal",
    "generate_candidates",
    "lag",
    "long_short_spread",
    "mean_reversion_signal",
    "momentum_signal",
    "rank",
    "ratio",
    "rolling_mean",
    "rolling_std",
    "rolling_sum",
    "screen_features",
    "sentiment_pressure_signal",
    "surprise_signal",
    "trend_signal",
    "volatility_signal",
    "volume_signal",
    "zscore",
]
