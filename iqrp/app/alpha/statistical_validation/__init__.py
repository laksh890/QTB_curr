"""Statistical validation for alpha research (IC, DSR, PBO, FDR)."""

from __future__ import annotations

# Lazy re-exports to avoid hard-failing on optional upstream package deps.

__all__ = [
    "ExperimentTracker",
    "block_bootstrap_ci",
    "deflated_sharpe_ratio",
    "false_discovery_report",
    "ic_significance",
    "iid_bootstrap_ci",
    "multiple_testing_adjustment",
    "newey_west_ic_significance",
    "permutation_ic_test",
    "probabilistic_sharpe_ratio",
    "probability_backtest_overfitting",
    "storey_qvalues",
]


def __getattr__(name: str):
    if name in {
        "block_bootstrap_ci",
        "iid_bootstrap_ci",
    }:
        from iqrp.app.alpha.statistical_validation import bootstrap as m

        return getattr(m, name)
    if name in {"deflated_sharpe_ratio", "probabilistic_sharpe_ratio"}:
        from iqrp.app.alpha.statistical_validation import deflated_sharpe as m

        return getattr(m, name)
    if name in {"false_discovery_report", "storey_qvalues"}:
        from iqrp.app.alpha.statistical_validation import false_discovery as m

        return getattr(m, name)
    if name in {"ExperimentTracker", "multiple_testing_adjustment"}:
        from iqrp.app.alpha.statistical_validation import multiple_testing as m

        return getattr(m, name)
    if name == "permutation_ic_test":
        from iqrp.app.alpha.statistical_validation.permutation import permutation_ic_test

        return permutation_ic_test
    if name == "probability_backtest_overfitting":
        from iqrp.app.alpha.statistical_validation.probability_backtest_overfitting import (
            probability_backtest_overfitting,
        )

        return probability_backtest_overfitting
    if name in {"ic_significance", "newey_west_ic_significance"}:
        from iqrp.app.alpha.statistical_validation import significance as m

        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
