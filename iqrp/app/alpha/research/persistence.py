"""Signal persistence / autocorrelation research diagnostics.

CRITICAL:
- Persistence of a signal is not proof of alpha.
- Statistical significance alone ≠ alpha.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from iqrp.app.alpha.discovery.symbolic import as_float1d, lag
from iqrp.app.features.research._numeric import pearson, safe_nanmean


def autocorrelation(signal: np.ndarray, lag_periods: int = 1) -> float:
    """Pearson autocorrelation at a given lag (PIT series vs its past)."""
    x = as_float1d(signal)
    if lag_periods < 1:
        raise ValueError("lag_periods must be >= 1")
    return float(pearson(x, lag(x, lag_periods)))


def persistence_profile(
    signal: np.ndarray,
    lags: Sequence[int] = (1, 2, 5, 10, 20),
) -> dict[str, Any]:
    """Autocorrelation across lags."""
    profile = {int(k): autocorrelation(signal, int(k)) for k in lags}
    vals = np.asarray(list(profile.values()), dtype=np.float64)
    return {
        "autocorr": profile,
        "mean_abs_autocorr": safe_nanmean(np.abs(vals)),
        "lag1": profile.get(1, float("nan")),
        "disclaimer": "Persistence ≠ alpha. Statistical significance alone ≠ alpha.",
    }


def signal_half_life(signal: np.ndarray, max_lag: int = 40) -> float:
    """Estimate AR(1)-style half-life from lag-1 autocorrelation.

    ``hl = -log(2) / log(|rho|)`` when 0 < |rho| < 1.
    """
    rho = autocorrelation(signal, 1)
    if not np.isfinite(rho) or abs(rho) < 1e-12 or abs(rho) >= 1.0:
        return float("nan")
    hl = float(-np.log(2.0) / np.log(abs(rho)))
    if hl > max_lag * 5:
        return float(max_lag * 5)
    return hl


def persistence_summary(signal: np.ndarray) -> dict[str, Any]:
    profile = persistence_profile(signal)
    return {
        **profile,
        "ar1_half_life": signal_half_life(signal),
    }
