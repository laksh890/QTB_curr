"""Stability of signal predictive power across regimes."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.alpha.regime.regime_performance import regime_ic, regime_performance


def regime_stability_score(
    signal: Any,
    forward_returns: Any,
    regimes: Any,
    *,
    min_obs: int = 20,
) -> dict[str, Any]:
    """Score how stable IC is across regimes.

    Score in [0, 1]: high when IC signs agree and magnitude dispersion is low.
    """
    rep = regime_ic(signal, forward_returns, regimes, rank=False)
    entries = []
    for name, payload in rep["by_regime"].items():
        n = int(payload.get("n_obs", payload.get("n_dates", 0)))
        ic = payload.get("ic", float("nan"))
        if n >= min_obs and np.isfinite(ic):
            entries.append((name, float(ic), n))

    if not entries:
        return {
            "name": "regime_stability_score",
            "score": 0.0,
            "n_regimes_used": 0,
            "sign_agreement": float("nan"),
            "ic_cv": float("nan"),
            "by_regime": rep["by_regime"],
        }

    ics = np.asarray([e[1] for e in entries], dtype=np.float64)
    signs = np.sign(ics)
    # fraction agreeing with majority sign (zeros ignored)
    nonzero = signs[signs != 0]
    if nonzero.size == 0:
        sign_agreement = 1.0
    else:
        majority = 1.0 if np.sum(nonzero > 0) >= np.sum(nonzero < 0) else -1.0
        sign_agreement = float(np.mean(nonzero == majority))

    mean_abs = float(np.mean(np.abs(ics)))
    ic_cv = float(np.std(ics) / (mean_abs + 1e-12))
    dispersion_penalty = float(np.clip(1.0 - ic_cv, 0.0, 1.0))
    score = float(np.clip(0.6 * sign_agreement + 0.4 * dispersion_penalty, 0.0, 1.0))

    return {
        "name": "regime_stability_score",
        "score": score,
        "n_regimes_used": len(entries),
        "sign_agreement": sign_agreement,
        "ic_cv": ic_cv,
        "mean_abs_ic": mean_abs,
        "by_regime": {e[0]: {"ic": e[1], "n": e[2]} for e in entries},
    }


def rolling_regime_stability(
    signal: Any,
    forward_returns: Any,
    regimes: Any,
    *,
    window: int = 60,
    step: int = 10,
) -> dict[str, Any]:
    """Rolling regime-stability scores over time."""
    sig = np.asarray(signal, dtype=np.float64)
    n = sig.shape[0]
    w = max(10, int(window))
    scores: list[float] = []
    indices: list[int] = []
    for start in range(0, max(0, n - w + 1), max(1, int(step))):
        sl = slice(start, start + w)
        if sig.ndim == 1:
            s, r = sig[sl], np.asarray(forward_returns, dtype=np.float64)[sl]
        else:
            s = sig[sl]
            r = np.asarray(forward_returns, dtype=np.float64)[sl]
        reg = np.asarray(regimes)[sl]
        rep = regime_stability_score(s, r, reg)
        scores.append(float(rep["score"]))
        indices.append(start + w - 1)
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "name": "rolling_regime_stability",
        "window": w,
        "step": int(step),
        "indices": indices,
        "scores": scores,
        "mean_score": float(np.nanmean(arr)) if arr.size else float("nan"),
        "min_score": float(np.nanmin(arr)) if arr.size else float("nan"),
    }


def regime_concentration(
    signal: Any,
    forward_returns: Any,
    regimes: Any,
) -> dict[str, Any]:
    """Detect if predictive power is concentrated in a single regime."""
    perf = regime_performance(signal, forward_returns, regimes)
    ics = []
    for name, payload in perf["ic"]["by_regime"].items():
        ic = payload.get("ic", float("nan"))
        if np.isfinite(ic):
            ics.append((name, abs(float(ic))))
    if not ics:
        return {
            "name": "regime_concentration",
            "herfindahl": float("nan"),
            "top_regime": None,
            "top_share": float("nan"),
            "concentrated": False,
        }
    vals = np.asarray([v for _, v in ics], dtype=np.float64)
    total = float(vals.sum()) + 1e-12
    shares = vals / total
    hhi = float(np.sum(shares**2))
    top_i = int(np.argmax(shares))
    return {
        "name": "regime_concentration",
        "herfindahl": hhi,
        "top_regime": ics[top_i][0],
        "top_share": float(shares[top_i]),
        "concentrated": bool(shares[top_i] >= 0.7 or hhi >= 0.6),
        "shares": {ics[i][0]: float(shares[i]) for i in range(len(ics))},
    }
