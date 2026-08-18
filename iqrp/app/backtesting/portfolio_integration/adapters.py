"""Thin adapters: Prompt 40 candidates → existing portfolio optimizers.

Causal alpha/cov only. Does not reimplement MV/RP/BL/HRP.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.portfolio.constraints import check_all_constraints
from iqrp.app.portfolio.constraints.concentration import concentration_metrics
from iqrp.app.portfolio.constraints.exposure import exposure_metrics
from iqrp.app.portfolio.constraints.turnover import turnover as turnover_fn
from iqrp.app.portfolio.covariance.sample import sample_covariance
from iqrp.app.portfolio.covariance.shrinkage import ledoit_wolf_covariance
from iqrp.app.portfolio.optimization import (
    optimize_black_litterman,
    optimize_hrp,
    optimize_mean_variance,
    optimize_risk_parity,
)


def daily_panel_from_series(
    series_map: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, dict[str, set]]]:
    """Build aligned daily net-return panel + per-candidate chronological period dates."""
    nets: dict[str, pd.Series] = {}
    period_dates: dict[str, dict[str, set]] = {}
    for cid, payload in series_map.items():
        d = payload["daily"]["net"]
        nets[cid] = d
        n = len(d)
        i_tr = max(int(n * 0.50), 1)
        i_va = max(int(n * 0.75), i_tr + 1)
        idx = list(d.index)
        period_dates[cid] = {
            "train": set(idx[:i_tr]),
            "validation": set(idx[i_tr:i_va]),
            "oos": set(idx[i_va:]),
            "pre_oos": set(idx[:i_va]),
            "full": set(idx),
        }
    panel = pd.DataFrame(nets).sort_index()
    return panel, period_dates


def causal_mu_cov(
    panel: pd.DataFrame,
    names: list[str],
    period_dates: dict[str, dict[str, set]],
    *,
    ridge: float = 1e-6,
) -> dict[str, Any]:
    """Estimate mu/cov on intersection of pre-OOS (train+val) dates — causal for OOS."""
    dates = set.intersection(*(period_dates[n]["pre_oos"] for n in names))
    idx = sorted(dates)
    if len(idx) < 10:
        # fallback: intersection of full calendars then take first 75%
        dates = set.intersection(*(period_dates[n]["full"] for n in names))
        idx_all = sorted(dates)
        cut = max(int(len(idx_all) * 0.75), 10)
        idx = idx_all[:cut]
    sub = panel.loc[idx, names].astype(float)
    # Drop rows that are all-nan; fill remaining nan with 0 for cov stability
    sub = sub.dropna(how="all").fillna(0.0)
    mu = sub.mean(axis=0).to_numpy(dtype=float)
    # Ledoit-Wolf preferred; fall back to sample + ridge
    try:
        lw = ledoit_wolf_covariance(sub.to_numpy(dtype=float))
        cov = np.asarray(lw["matrix"], dtype=float)
        cov_method = "ledoit_wolf"
    except Exception:  # noqa: BLE001
        sc = sample_covariance(sub.to_numpy(dtype=float))
        cov = np.asarray(sc["matrix"], dtype=float)
        cov_method = "sample"
    n = cov.shape[0]
    cov = 0.5 * (cov + cov.T) + ridge * np.eye(n)
    # Degenerate check
    eig = np.linalg.eigvalsh(cov)
    degenerate = bool(np.min(eig) < 1e-12)
    if degenerate:
        cov = cov + max(ridge, 1e-4) * np.eye(n)
    return {
        "mu": mu,
        "cov": cov,
        "names": names,
        "n_obs": int(len(sub)),
        "cov_method": cov_method,
        "degenerate_before_ridge": degenerate,
        "min_eigenvalue": float(np.min(np.linalg.eigvalsh(cov))),
        "estimation_window": "pre_oos_train_plus_validation",
        "n_dates": len(idx),
    }


def run_optimizer(
    method: str,
    *,
    mu: np.ndarray,
    cov: np.ndarray,
    names: list[str],
    max_weight: float,
    max_gross: float,
    budget: float,
    risk_aversion: float,
    long_only_sleeves: bool,
) -> dict[str, Any]:
    """Call existing optimizer APIs. No reimplementation."""
    # Long/short MV/BL with sum(w)=budget and max_gross≈|budget| is often infeasible.
    # When long_only_sleeves=False, use dollar-neutral budget=0 so gross can hold L/S.
    eff_budget = float(budget)
    eff_gross = float(max_gross)
    if method in {"mean_variance", "black_litterman"} and not long_only_sleeves:
        eff_budget = 0.0
        eff_gross = max(float(max_gross), 1.0)

    common = dict(
        cov=cov,
        names=names,
        max_weight=max_weight,
        max_gross=eff_gross,
        budget=eff_budget,
    )
    if method == "mean_variance":
        return optimize_mean_variance(
            mu=mu,
            long_only=long_only_sleeves,
            risk_aversion=risk_aversion,
            min_weight=(-max_weight if not long_only_sleeves else 0.0),
            **common,
        )
    if method == "risk_parity":
        # RP is long-only on sleeve budgets by design of existing solver
        return optimize_risk_parity(mu=mu, long_only=True, max_weight=max_weight, max_gross=max_gross, budget=budget, cov=cov, names=names)
    if method == "black_litterman":
        n = len(names)
        P = np.eye(n)
        Q = mu.copy()
        eq = np.zeros(n)
        return optimize_black_litterman(
            mu=eq,
            long_only=long_only_sleeves,
            risk_aversion=risk_aversion,
            market_weights=np.ones(n) / n,
            P=P,
            Q=Q,
            tau=0.05,
            min_weight=(-max_weight if not long_only_sleeves else 0.0),
            **common,
        )
    if method == "hrp":
        return optimize_hrp(mu=mu, long_only=True, max_weight=max_weight, max_gross=max_gross, budget=budget, cov=cov, names=names)
    if method == "constraints_only":
        w = np.ones(len(names)) / max(len(names), 1)
        if w.max() > max_weight + 1e-12:
            w = np.minimum(w, max_weight)
            w = w / w.sum() * budget
        return {
            "name": "constraints_only",
            "method": "equal_sleeve_budget",
            "success": True,
            "weights": {names[i]: float(w[i]) for i in range(len(names))},
            "status": "OK",
            "diagnostics": {"note": "Baseline without PortfolioOptimizer objective"},
        }
    raise ValueError(f"unknown method {method}")


def weights_dict(opt_result: dict[str, Any], names: list[str]) -> dict[str, float]:
    w = opt_result.get("weights")
    if isinstance(w, dict):
        return {k: float(w.get(k, 0.0)) for k in names}
    if isinstance(w, (list, tuple, np.ndarray)):
        arr = np.asarray(w, dtype=float).reshape(-1)
        return {names[i]: float(arr[i]) for i in range(len(names))}
    return {n: 0.0 for n in names}


def apply_direction_to_sleeves(
    sleeve_weights: dict[str, float],
    directions: dict[str, str],
) -> dict[str, float]:
    """Map non-negative (or signed) sleeve budgets to signed exposure via candidate direction.

    LONG → +w, SHORT → -w, LONG_SHORT → +w (signal sign applied later bar-by-bar).
    Static direction metadata used for portfolio-level exposure reporting when signal
    is not yet applied; bar-level signing uses live signal series.
    """
    out: dict[str, float] = {}
    for cid, w in sleeve_weights.items():
        d = directions.get(cid, "LONG_SHORT")
        if d == "SHORT":
            out[cid] = -abs(float(w))
        elif d == "LONG":
            out[cid] = abs(float(w))
        else:
            # LONG_SHORT: keep magnitude; sign comes from signal each bar
            out[cid] = abs(float(w))
    return out


def signed_exposure_series(
    sleeve_weights: dict[str, float],
    signal_signs: dict[str, pd.Series],
    directions: dict[str, str],
    index: pd.Index,
) -> pd.DataFrame:
    """Bar-aligned signed exposures = sleeve_budget * direction_policy * signal_sign."""
    cols = {}
    for cid, w in sleeve_weights.items():
        mag = abs(float(w))
        sig = signal_signs.get(cid)
        if sig is None:
            cols[cid] = pd.Series(0.0, index=index)
            continue
        s = sig.reindex(index).fillna(0.0)
        d = directions.get(cid, "LONG_SHORT")
        if d == "LONG":
            signed = mag * np.clip(np.sign(s.to_numpy()), 0, None)
        elif d == "SHORT":
            # SHORT candidates: positive sleeve * negative signal region → short exposure
            # Prefer candidate already direction-masked; use -mag when signal short/nonzero
            raw = np.sign(s.to_numpy())
            # If signal already short-only, sign is -1 or 0; exposure = mag * sign (negative)
            # If signal is ±1 LONG_SHORT masked to SHORT upstream, same.
            signed = mag * np.where(raw < 0, -1.0, np.where(raw > 0, -1.0, 0.0))
            # For SHORT direction metadata with already-masked short signals (values ≤0):
            signed = mag * np.where(np.abs(raw) > 0, -1.0, 0.0)
        else:
            signed = mag * np.sign(s.to_numpy())
        cols[cid] = pd.Series(signed, index=index)
    return pd.DataFrame(cols)


def validate_weights(
    weights: dict[str, float],
    *,
    max_weight: float,
    max_gross: float,
    max_net: float,
    max_turnover: float,
    previous: dict[str, float] | None = None,
    adv: dict[str, float] | None = None,
) -> dict[str, Any]:
    names = list(weights.keys())
    w = np.array([weights[n] for n in names], dtype=float)
    prev = np.array([float((previous or {}).get(n, 0.0)) for n in names], dtype=float)
    kwargs: dict[str, Any] = {
        "names": names,
        "max_weight": max_weight,
        "max_gross": max_gross,
        "max_net": max_net,
        "max_turnover": max_turnover,
        "current_weights": prev,
    }
    if adv is not None:
        kwargs["adv"] = np.array([float(adv.get(n, 1e12)) for n in names], dtype=float)
        kwargs["max_participation"] = 0.5
    violations = check_all_constraints(w, **kwargs)
    exp = exposure_metrics(w)
    conc = concentration_metrics(w)
    to = float(turnover_fn(prev, w))
    return {
        "violations": [
            {
                "name": getattr(v, "name", str(v)),
                "severity": str(getattr(v, "severity", "")),
                "message": getattr(v, "message", str(v)),
            }
            for v in (violations or [])
        ],
        "n_violations": len(violations or []),
        "exposure": exp if isinstance(exp, dict) else {"raw": str(exp)},
        "concentration": conc if isinstance(conc, dict) else {"raw": str(conc)},
        "turnover_vs_previous": to,
        "active_positions": int(np.sum(np.abs(w) > 1e-8)),
        "long_exposure": float(np.sum(w[w > 0])) if np.any(w > 0) else 0.0,
        "short_exposure": float(np.sum(w[w < 0])) if np.any(w < 0) else 0.0,
        "gross_exposure": float(np.sum(np.abs(w))),
        "net_exposure": float(np.sum(w)),
    }


__all__ = [
    "apply_direction_to_sleeves",
    "causal_mu_cov",
    "daily_panel_from_series",
    "run_optimizer",
    "signed_exposure_series",
    "validate_weights",
    "weights_dict",
]
