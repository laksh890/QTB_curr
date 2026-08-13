"""Hard constraint projection for capital weights.

Capital allocation NEVER overrides hard risk limits — always clip to
portfolio/strategy caps from settings/constraints.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.capital.config import CapitalSettings


def project_weights(
    weights: np.ndarray | list[float],
    *,
    settings: CapitalSettings | None = None,
    max_weight: float | None = None,
    min_weight: float | None = None,
    max_gross: float | None = None,
    max_concentration: float | None = None,
    max_leverage: float | None = None,
    n_iter: int = 50,
) -> dict[str, Any]:
    """Project weights onto hard box / simplex / concentration / leverage constraints."""
    cfg = settings or CapitalSettings.default()
    w = np.asarray(weights, dtype=np.float64).ravel()
    n = w.size
    applied: list[str] = []
    if n == 0:
        return {"weights": w, "constraints_applied": applied, "feasible": True}

    lo = float(cfg.min_weight if min_weight is None else min_weight)
    hi = float(cfg.max_weight if max_weight is None else max_weight)
    conc = float(cfg.max_concentration if max_concentration is None else max_concentration)
    hi = min(hi, conc)
    gross_cap = float(cfg.max_gross_exposure if max_gross is None else max_gross)
    lev_cap = float(cfg.max_leverage if max_leverage is None else max_leverage)

    w = np.maximum(w, 0.0)
    # Preserve intentional zero mass (e.g. TRADING_HALT) — never resurrect capital
    if float(np.sum(w)) <= 1e-18:
        applied.append("zero_mass_halt")
        return {
            "weights": np.zeros(n, dtype=np.float64),
            "constraints_applied": applied,
            "feasible": False,
            "gross": 0.0,
            "max_weight": 0.0,
        }

    # Iterative box + simplex projection
    for _ in range(max(int(n_iter), 1)):
        w = np.clip(w, lo, hi)
        applied.append("box_clip")
        s = float(np.sum(w))
        if s <= 1e-18:
            # All mass removed by hard caps — remain at zero (do not invent weights)
            w = np.zeros(n, dtype=np.float64)
            applied.append("zero_after_box_clip")
            break
        # Target sum = min(1, gross_cap) for long-only capital weights
        target = min(1.0, gross_cap)
        w = w * (target / s)
        applied.append("simplex_renorm")
        if float(np.max(w)) <= hi + 1e-12 and float(np.min(w[w > 0])) >= lo - 1e-12:
            break

    # Gross / leverage soft check (long-only gross = sum)
    gross = float(np.sum(np.abs(w)))
    if gross > gross_cap + 1e-12:
        w = w * (gross_cap / gross)
        applied.append("gross_cap")
    if gross > lev_cap + 1e-12:
        w = w * (lev_cap / max(gross, 1e-12))
        applied.append("leverage_cap")

    # Concentration: HHI soft shrink of largest names if needed
    if n > 0 and float(np.max(w)) > conc + 1e-12:
        w = np.minimum(w, conc)
        s = float(np.sum(w))
        if s > 0:
            w = w / s * min(1.0, gross_cap)
        applied.append("concentration_cap")

    # Deduplicate applied labels while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for a in applied:
        if a not in seen:
            seen.add(a)
            uniq.append(a)

    return {
        "weights": w,
        "constraints_applied": uniq,
        "feasible": bool(float(np.sum(w)) > 1e-12 or n == 0),
        "gross": float(np.sum(np.abs(w))),
        "max_weight": float(np.max(w)) if n else 0.0,
    }


def apply_turnover_constraint(
    current: np.ndarray | list[float],
    target: np.ndarray | list[float],
    *,
    max_turnover: float = 0.50,
) -> dict[str, Any]:
    """Move from current toward target without exceeding turnover (L1/2)."""
    c = np.asarray(current, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    n = min(c.size, t.size)
    if n == 0:
        return {"weights": t, "turnover": 0.0, "scaled": False}
    c = c[:n]
    t = t[:n]
    delta = t - c
    turnover = 0.5 * float(np.sum(np.abs(delta)))
    cap = max(float(max_turnover), 0.0)
    scaled = False
    if turnover > cap + 1e-12 and turnover > 0:
        delta = delta * (cap / turnover)
        scaled = True
        turnover = cap
    w = c + delta
    w = np.maximum(w, 0.0)
    s = float(np.sum(w))
    if s > 0:
        w = w / s
    return {"weights": w, "turnover": turnover, "scaled": scaled}


def apply_participation_constraint(
    weights: np.ndarray | list[float],
    *,
    capital: float,
    adv: np.ndarray | list[float] | None,
    max_participation: float = 0.10,
    ttl_days: float = 5.0,
) -> dict[str, Any]:
    """Downscale weights whose notional would exceed ADV * participation * TTL."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    n = w.size
    if adv is None or n == 0:
        return {"weights": w, "scaled": False, "participation": []}
    a = np.asarray(adv, dtype=np.float64).ravel()
    if a.size != n:
        return {"weights": w, "scaled": False, "participation": []}
    cap = max(float(capital), 0.0)
    part_cap = float(np.clip(max_participation, 1e-6, 1.0))
    ttl = max(float(ttl_days), 1e-6)
    max_notional = part_cap * np.maximum(a, 1e-12) * ttl
    notionals = cap * np.maximum(w, 0.0)
    scales = np.ones(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(max_notional > 0, notionals / max_notional, np.inf)
    over = ratio > 1.0
    scales[over] = 1.0 / ratio[over]
    scaled = bool(np.any(over))
    out = np.maximum(w, 0.0) * scales
    s = float(np.sum(out))
    if s > 0:
        out = out / s
    return {
        "weights": out,
        "scaled": scaled,
        "participation": (notionals / np.maximum(a, 1e-12)).tolist(),
        "scales": scales.tolist(),
    }
