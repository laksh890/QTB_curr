"""IC / hit-rate decay across forward horizons.

CRITICAL:
- Decay analysis informs holding-period research; it does not approve alpha.
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve.
- Forward returns for horizon h use future data as *targets* only — the signal
  itself must already be point-in-time (past windows only).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from iqrp.app.alpha.research.hit_rate import compute_hit_rate
from iqrp.app.alpha.research.information_coefficient import compute_ic
from iqrp.app.alpha.research.rank_ic import compute_rank_ic


def forward_returns(returns: np.ndarray, horizon: int) -> np.ndarray:
    """Sum of future returns over ``horizon`` bars: ``r[t+1] + ... + r[t+horizon]``.

    This is a *target* construction (labels), not a signal. Trailing NaNs mark
    unavailable future windows.
    """
    r = np.asarray(returns, dtype=np.float64)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    n = len(r)
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(n - horizon):
        window = r[t + 1 : t + 1 + horizon]
        if np.isfinite(window).sum() < horizon:
            continue
        out[t] = float(np.sum(window))
    return out


def _half_life_from_ics(horizons: Sequence[int], ics: Sequence[float]) -> float:
    """Estimate horizon where |IC| falls to half of the first finite |IC|.

    Uses log-linear interpolation between adjacent horizons when possible.
    Returns NaN if decay cannot be estimated.
    """
    hs = list(horizons)
    abs_ics = [abs(v) if np.isfinite(v) else float("nan") for v in ics]
    base = next((v for v in abs_ics if np.isfinite(v) and v > 1e-12), None)
    if base is None:
        return float("nan")
    target = 0.5 * base
    # If already below target at first point
    first_finite_idx = next(i for i, v in enumerate(abs_ics) if np.isfinite(v))
    if abs_ics[first_finite_idx] <= target:
        return float(hs[first_finite_idx])
    for i in range(first_finite_idx, len(abs_ics) - 1):
        a, b = abs_ics[i], abs_ics[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a >= target >= b or a <= target <= b:
            # linear interp in horizon space
            if abs(a - b) < 1e-15:
                return float(hs[i])
            w = (a - target) / (a - b)
            return float(hs[i] + w * (hs[i + 1] - hs[i]))
    # Extrapolate exponential decay from first two finite points if available
    finite_pairs = [(hs[i], abs_ics[i]) for i in range(len(hs)) if np.isfinite(abs_ics[i])]
    if len(finite_pairs) >= 2:
        (h0, y0), (h1, y1) = finite_pairs[0], finite_pairs[1]
        if y0 > 1e-12 and y1 > 1e-12 and y1 < y0:
            # y = y0 * exp(-lambda*(h-h0)); half when exp(-lambda*dh)=0.5
            lam = -np.log(y1 / y0) / max(h1 - h0, 1e-12)
            if lam > 0:
                return float(h0 + np.log(2.0) / lam)
    # Never crossed half within grid → beyond last horizon
    return float(hs[-1]) if hs else float("nan")


def analyze_decay(
    signal: np.ndarray,
    returns: np.ndarray,
    horizons: Sequence[int] = (1, 2, 5, 10),
) -> dict[str, Any]:
    """IC and hit-rate across horizons; half-life and optimal hold.

    ``returns`` are period returns used to build forward targets.
    Signal must already be PIT (past-only).
    """
    x = np.asarray(signal, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    if len(x) != len(r):
        raise ValueError("signal and returns length mismatch")
    hs = [int(h) for h in horizons]
    if not hs or any(h < 1 for h in hs):
        raise ValueError("horizons must be positive integers")

    ic_by_h: dict[int, float] = {}
    rank_ic_by_h: dict[int, float] = {}
    hit_by_h: dict[int, float] = {}
    for h in hs:
        fwd = forward_returns(r, h)
        ic_by_h[h] = compute_ic(x, fwd)
        rank_ic_by_h[h] = compute_rank_ic(x, fwd)
        hit_by_h[h] = compute_hit_rate(x, fwd)

    ics = [ic_by_h[h] for h in hs]
    half_life = _half_life_from_ics(hs, ics)

    # Optimal hold = horizon with max |IC|
    best_h = hs[0]
    best_abs = -1.0
    for h in hs:
        v = abs(ic_by_h[h]) if np.isfinite(ic_by_h[h]) else -1.0
        if v > best_abs:
            best_abs = v
            best_h = h

    return {
        "horizons": hs,
        "ic": ic_by_h,
        "rank_ic": rank_ic_by_h,
        "hit_rate": hit_by_h,
        "half_life": half_life,
        "optimal_hold": best_h,
        "disclaimer": (
            "Decay diagnostics inform research holding periods. "
            "Statistical significance alone ≠ alpha. "
            "Historical Sharpe alone cannot approve."
        ),
    }
