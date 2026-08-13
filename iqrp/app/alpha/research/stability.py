"""Rolling IC stability for alpha research.

CRITICAL:
- Stability of IC is a research diagnostic, not approval.
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.alpha.research.decay import forward_returns
from iqrp.app.alpha.research.information_coefficient import compute_ic, rolling_ic
from iqrp.app.features.research._numeric import safe_nanmean


def analyze_stability(
    signal: np.ndarray,
    returns: np.ndarray,
    *,
    horizon: int = 1,
    window: int = 60,
    step: int = 10,
    min_obs: int = 30,
) -> dict[str, Any]:
    """Rolling IC stability vs forward returns at a fixed horizon."""
    x = np.asarray(signal, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    fwd = forward_returns(r, horizon)
    roll = rolling_ic(x, fwd, window=window, step=step, min_obs=min_obs)
    finite = roll[np.isfinite(roll)]
    mean = safe_nanmean(roll)
    std = float(np.std(finite)) if finite.size else float("nan")
    ir = float(mean / (std + 1e-12)) if finite.size else float("nan")
    # Fraction of windows with same sign as overall IC
    overall = compute_ic(x, fwd)
    if finite.size and np.isfinite(overall) and overall != 0:
        sign_consistency = float(np.mean(np.sign(finite) == np.sign(overall)))
    else:
        sign_consistency = float("nan")
    # Stability score in [0, 1]: high mean/std and sign consistency
    if finite.size >= 3 and np.isfinite(std):
        cv_pen = 1.0 / (1.0 + abs(std) / (abs(mean) + 1e-9))
        stab = float(np.clip(0.5 * cv_pen + 0.5 * (sign_consistency if np.isfinite(sign_consistency) else 0.0), 0, 1))
    else:
        stab = float("nan")
    return {
        "overall_ic": overall,
        "rolling_ic_mean": mean,
        "rolling_ic_std": std,
        "rolling_ic_ir": ir,
        "sign_consistency": sign_consistency,
        "stability_score": stab,
        "n_windows": int(finite.size),
        "window": window,
        "step": step,
        "horizon": horizon,
        "disclaimer": (
            "Rolling IC stability ≠ alpha. "
            "Historical Sharpe alone cannot approve."
        ),
    }
