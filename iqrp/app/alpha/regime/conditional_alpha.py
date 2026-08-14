"""Conditional alpha: performance given regime / state filters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from iqrp.app.alpha.regime.regime_performance import regime_ic


def conditional_ic(
    signal: Any,
    forward_returns: Any,
    condition: Any,
    *,
    rank: bool = False,
) -> dict[str, Any]:
    """IC computed only on observations where ``condition`` is truthy."""
    sig = np.asarray(signal, dtype=np.float64)
    ret = np.asarray(forward_returns, dtype=np.float64)
    cond = np.asarray(condition).astype(bool)
    if cond.shape[0] != sig.shape[0]:
        raise ValueError("condition length must match time dimension")

    if sig.ndim == 1:
        s, r = sig[cond], ret[cond]
        m = np.isfinite(s) & np.isfinite(r)
        if m.sum() < 3:
            ic = float("nan")
        else:
            a, b = s[m], r[m]
            if rank:
                a = a.argsort().argsort().astype(np.float64)
                b = b.argsort().argsort().astype(np.float64)
            ic = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
        return {
            "name": "conditional_ic",
            "ic": ic,
            "n_obs": int(cond.sum()),
            "coverage": float(cond.mean()) if cond.size else 0.0,
        }

    ics = []
    for i in np.where(cond)[0]:
        s, r = sig[i], ret[i]
        m = np.isfinite(s) & np.isfinite(r)
        if m.sum() < 3:
            continue
        a, b = s[m], r[m]
        if rank:
            a = a.argsort().argsort().astype(np.float64)
            b = b.argsort().argsort().astype(np.float64)
        if np.std(a) < 1e-15 or np.std(b) < 1e-15:
            continue
        ics.append(float(np.corrcoef(a, b)[0, 1]))
    arr = np.asarray(ics, dtype=np.float64)
    return {
        "name": "conditional_ic",
        "ic": float(np.nanmean(arr)) if arr.size else float("nan"),
        "ic_std": float(np.nanstd(arr)) if arr.size else float("nan"),
        "n_dates": int(cond.sum()),
        "coverage": float(cond.mean()) if cond.size else 0.0,
    }


def conditional_alpha_profile(
    signal: Any,
    forward_returns: Any,
    conditions: Mapping[str, Any],
) -> dict[str, Any]:
    """IC under multiple named conditions (e.g. high_vol, bull, liquid)."""
    profiles: dict[str, Any] = {}
    for name, cond in conditions.items():
        profiles[str(name)] = conditional_ic(signal, forward_returns, cond)
    ics = [p["ic"] for p in profiles.values() if np.isfinite(p.get("ic", np.nan))]
    return {
        "name": "conditional_alpha_profile",
        "conditions": profiles,
        "best_condition": (
            max(profiles.items(), key=lambda kv: abs(kv[1].get("ic") or 0.0))[0]
            if profiles
            else None
        ),
        "ic_range": float(np.ptp(ics)) if ics else float("nan"),
    }


def regime_gated_signal(
    signal: Any,
    regimes: Any,
    active_regimes: set[Any] | list[Any],
    *,
    inactive_value: float = 0.0,
) -> np.ndarray:
    """Zero (or set) the signal outside allowed regimes."""
    sig = np.asarray(signal, dtype=np.float64).copy()
    labels = np.asarray(regimes)
    if labels.shape[0] != sig.shape[0]:
        raise ValueError("regimes length mismatch")
    active = {str(x) for x in active_regimes}
    mask = np.asarray([str(x) in active for x in labels.tolist()])
    if sig.ndim == 1:
        sig[~mask] = inactive_value
    else:
        sig[~mask, :] = inactive_value
    return sig


def compare_unconditional_vs_conditional(
    signal: Any,
    forward_returns: Any,
    regimes: Any,
) -> dict[str, Any]:
    """Side-by-side unconditional IC vs per-regime IC."""
    sig = np.asarray(signal, dtype=np.float64)
    ret = np.asarray(forward_returns, dtype=np.float64)
    if sig.ndim == 1:
        m = np.isfinite(sig) & np.isfinite(ret)
        unc = (
            float(np.corrcoef(sig[m], ret[m])[0, 1])
            if m.sum() >= 3 and np.std(sig[m]) > 0 and np.std(ret[m]) > 0
            else float("nan")
        )
    else:
        daily = []
        for i in range(sig.shape[0]):
            m = np.isfinite(sig[i]) & np.isfinite(ret[i])
            if m.sum() < 3:
                continue
            a, b = sig[i][m], ret[i][m]
            if np.std(a) < 1e-15 or np.std(b) < 1e-15:
                continue
            daily.append(float(np.corrcoef(a, b)[0, 1]))
        unc = float(np.nanmean(daily)) if daily else float("nan")

    cond = regime_ic(sig, ret, regimes)
    return {
        "name": "unconditional_vs_conditional",
        "unconditional_ic": unc,
        "conditional": cond,
        "lift": {
            k: (
                (float(v["ic"]) - unc)
                if np.isfinite(v.get("ic", np.nan)) and np.isfinite(unc)
                else float("nan")
            )
            for k, v in cond["by_regime"].items()
        },
    }


def apply_condition_fn(
    signal: Any,
    forward_returns: Any,
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, Any]:
    """Apply a custom boolean mask factory ``fn(signal, returns) -> mask``."""
    sig = np.asarray(signal, dtype=np.float64)
    ret = np.asarray(forward_returns, dtype=np.float64)
    mask = np.asarray(fn(sig, ret)).astype(bool)
    return conditional_ic(sig, ret, mask)
