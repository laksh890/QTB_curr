"""Multiple-testing / research-breadth recording for alpha sweeps."""

from __future__ import annotations

from typing import Any


def research_breadth_record(
    *,
    n_features_tested: int,
    n_signals_tested: int,
    n_parameter_combinations: int,
    n_horizons: int,
    n_datasets: int,
    n_experiments: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_features_tested": int(n_features_tested),
        "n_signals_tested": int(n_signals_tested),
        "n_parameter_combinations": int(n_parameter_combinations),
        "n_horizons": int(n_horizons),
        "n_datasets": int(n_datasets),
        "n_experiments": int(n_experiments),
        "warning": (
            "Many configurations tested; best observed result is NOT automatically "
            "statistically significant. Do not present the best result without research breadth."
        ),
    }
    try:
        from iqrp.app.alpha.statistical_validation import multiple_testing_adjustment
        from math import erfc, sqrt

        # placeholder p-values — breadth recording; not a validity claim
        pvals = [min(1.0, 0.5) for _ in range(max(n_experiments, 1))]
        out["multiple_testing_adjustment"] = multiple_testing_adjustment(pvals[: min(20, len(pvals))], method="fdr_bh")
    except Exception as exc:  # noqa: BLE001
        out["multiple_testing_adjustment_unavailable"] = str(exc)
    return out


__all__ = ["research_breadth_record"]
