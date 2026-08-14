"""Multiple-testing awareness for horizon sweeps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def multiple_testing_record(
    *,
    n_configurations: int,
    n_strategies: int = 1,
    n_horizons: int = 1,
    n_parameter_combinations: int = 1,
    observed_sharpes: Sequence[float] | None = None,
    returns_for_best: Any | None = None,
) -> dict[str, Any]:
    """Record trial counts and optional deflated-Sharpe / corrections.

    Does not treat the best observed result as automatically significant.
    """
    out: dict[str, Any] = {
        "n_configurations_tested": int(n_configurations),
        "n_strategies_tested": int(n_strategies),
        "n_horizons_tested": int(n_horizons),
        "n_parameter_combinations_tested": int(n_parameter_combinations),
        "warning": (
            "Many horizons/configs tested; best observed result is NOT automatically "
            "statistically significant."
        ),
    }
    sharpes = [float(x) for x in (observed_sharpes or [])]
    if sharpes:
        out["best_observed_sharpe"] = float(max(sharpes))
        out["median_observed_sharpe"] = float(np.median(sharpes))
        out["n_positive_sharpe"] = int(sum(1 for s in sharpes if s > 0))

    # Optional integration with existing statistical validation
    try:
        from iqrp.app.alpha.statistical_validation import deflated_sharpe_ratio
        from iqrp.app.backtesting.performance.risk_adjusted import sharpe_ratio

        if returns_for_best is not None and n_configurations > 1:
            r = np.asarray(returns_for_best, dtype=np.float64).reshape(-1)
            if r.size >= 5:
                obs = float(sharpe_ratio(r, periods_per_year=1.0))  # non-annualized mean/std
                dsr = deflated_sharpe_ratio(
                    obs,
                    n_trials=max(int(n_configurations), 1),
                    n_obs=int(r.size),
                    return_details=True,
                )
                if isinstance(dsr, Mapping):
                    out["deflated_sharpe"] = dsr.get("deflated_sharpe", dsr.get("dsr"))
                    out["deflated_sharpe_detail"] = dict(dsr)
                else:
                    out["deflated_sharpe"] = float(dsr)
    except Exception as exc:  # noqa: BLE001
        out["deflated_sharpe_unavailable"] = str(exc)

    try:
        from iqrp.app.alpha.statistical_validation import multiple_testing_adjustment

        if sharpes:
            # convert crude z-like scores from sharpes for illustration only
            pvals = []
            for s in sharpes:
                # two-sided normal approx under null sharpe=0, se~1 (conservative placeholder)
                from math import erfc, sqrt

                z = abs(s)  # already annualized-ish; treat cautiously
                p = float(erfc(z / sqrt(2.0)))
                pvals.append(min(max(p, 1e-12), 1.0))
            adj = multiple_testing_adjustment(pvals, method="fdr_bh")
            out["multiple_testing_adjustment"] = adj if isinstance(adj, dict) else {"adjusted": adj}
    except Exception as exc:  # noqa: BLE001
        out["multiple_testing_adjustment_unavailable"] = str(exc)

    return out


__all__ = ["multiple_testing_record"]
