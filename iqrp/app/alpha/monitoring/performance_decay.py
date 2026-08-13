"""Performance decay monitoring (returns / Sharpe / drawdown)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _sharpe(x: np.ndarray, *, ann: float = 252.0) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if sd < 1e-15:
        return float("nan")
    return float(np.mean(x) / sd * np.sqrt(ann))


def rolling_performance(
    returns: Any,
    *,
    window: int = 60,
    step: int = 5,
    ann_factor: float = 252.0,
) -> dict[str, Any]:
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    w = max(5, int(window))
    sharpes: list[float] = []
    means: list[float] = []
    indices: list[int] = []
    for start in range(0, max(0, r.size - w + 1), max(1, int(step))):
        sl = r[start : start + w]
        means.append(float(np.nanmean(sl)))
        sharpes.append(_sharpe(sl, ann=ann_factor))
        indices.append(start + w - 1)
    return {
        "name": "rolling_performance",
        "window": w,
        "step": int(step),
        "indices": indices,
        "mean_return": means,
        "sharpe": sharpes,
        "last_sharpe": float(sharpes[-1]) if sharpes else float("nan"),
        "last_mean": float(means[-1]) if means else float("nan"),
    }


def performance_decay_score(
    returns: Any,
    *,
    baseline_window: int = 120,
    recent_window: int = 40,
) -> dict[str, Any]:
    """Compare recent vs baseline Sharpe/mean; score in [0,1] (1 = no decay)."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    bw = max(10, int(baseline_window))
    rw = max(5, int(recent_window))
    if r.size < bw + 1:
        return {
            "name": "performance_decay_score",
            "score": 0.0,
            "baseline_sharpe": float("nan"),
            "recent_sharpe": float("nan"),
            "decayed": True,
        }
    # baseline: earlier portion; recent: last rw
    recent = r[-rw:]
    base = r[-(bw + rw) : -rw] if r.size >= bw + rw else r[:-rw]
    sh_b = _sharpe(base)
    sh_r = _sharpe(recent)
    mu_b = float(np.mean(base)) if base.size else float("nan")
    mu_r = float(np.mean(recent)) if recent.size else float("nan")

    if np.isfinite(sh_b) and abs(sh_b) > 1e-9 and np.isfinite(sh_r):
        ratio = sh_r / sh_b if sh_b > 0 else (0.0 if sh_r < 0 else 1.0)
    else:
        ratio = 0.0 if (np.isfinite(mu_r) and mu_r < 0) else float("nan")

    score = float(np.clip(ratio, 0.0, 1.0)) if np.isfinite(ratio) else 0.0
    decayed = score < 0.5 or (np.isfinite(sh_r) and sh_r < 0 and np.isfinite(sh_b) and sh_b > 0)
    return {
        "name": "performance_decay_score",
        "score": score,
        "baseline_sharpe": sh_b,
        "recent_sharpe": sh_r,
        "baseline_mean": mu_b,
        "recent_mean": mu_r,
        "decayed": bool(decayed),
    }


def max_drawdown(returns: Any) -> float:
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    r = np.where(np.isfinite(r), r, 0.0)
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(dd)) if dd.size else 0.0


def monitor_performance_decay(
    returns: Any,
    *,
    baseline_sharpe: float | None = None,
    sharpe_floor: float = 0.0,
) -> dict[str, Any]:
    dec = performance_decay_score(returns)
    recent_sh = dec["recent_sharpe"]
    base = float(baseline_sharpe) if baseline_sharpe is not None else dec["baseline_sharpe"]
    status = "HEALTHY"
    if dec["decayed"] or (np.isfinite(recent_sh) and recent_sh < sharpe_floor):
        status = "DEGRADED"
    if np.isfinite(recent_sh) and recent_sh < 0 and np.isfinite(base) and base > 0.5:
        status = "COLLAPSED"
    return {
        "name": "monitor_performance_decay",
        "status": status,
        "decay": dec,
        "max_drawdown": max_drawdown(returns),
        "baseline_sharpe": base,
    }
