"""Regime-conditioned alpha analysis."""

from __future__ import annotations

from iqrp.app.alpha.regime.conditional_alpha import (
    apply_condition_fn,
    compare_unconditional_vs_conditional,
    conditional_alpha_profile,
    conditional_ic,
    regime_gated_signal,
)
from iqrp.app.alpha.regime.regime_performance import (
    regime_hit_rate,
    regime_ic,
    regime_performance,
    regime_returns,
)
from iqrp.app.alpha.regime.regime_stability import (
    regime_concentration,
    regime_stability_score,
    rolling_regime_stability,
)

__all__ = [
    "apply_condition_fn",
    "compare_unconditional_vs_conditional",
    "conditional_alpha_profile",
    "conditional_ic",
    "regime_concentration",
    "regime_gated_signal",
    "regime_hit_rate",
    "regime_ic",
    "regime_performance",
    "regime_returns",
    "regime_stability_score",
    "rolling_regime_stability",
]
