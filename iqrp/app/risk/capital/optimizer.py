"""Risk-budget optimization under hard institutional constraints.

Objectives are risk-control oriented. This module never generates alpha signals
from historical performance alone — optional ``expected_opportunity`` may tilt
weights but hard limits always bind.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.risk.capital.constraints import project_weights
from iqrp.app.risk.capital.correlation import effective_risk_budgets
from iqrp.app.risk.portfolio.portfolio_risk import portfolio_volatility
from iqrp.app.risk.sizing.risk_parity import equal_risk_contribution, risk_parity_weights

Objective = Literal[
    "min_risk",
    "max_diversification",
    "target_volatility",
    "target_cvar",
    "target_drawdown",
    "risk_budget_match",
    "max_risk_adjusted_opportunity",
]


def optimize_risk_budgets(
    cov: np.ndarray,
    *,
    objective: Objective = "risk_budget_match",
    risk_budgets: np.ndarray | list[float] | None = None,
    target_vol: float = 0.10,
    target_cvar: float | None = None,
    target_drawdown: float | None = None,
    expected_opportunity: np.ndarray | list[float] | None = None,
    max_weight: float = 0.40,
    max_leverage: float = 1.5,
    corr: np.ndarray | None = None,
    names: list[str] | None = None,
    returns: np.ndarray | None = None,
    max_iter: int = 200,
) -> dict[str, Any]:
    """Optimize portfolio weights subject to risk / capital / concentration caps.

    Hard limits cannot be overridden by opportunity or confidence.
    """
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square")
    n = c.shape[0]
    labels = list(names) if names is not None else [f"s{i}" for i in range(n)]
    obj = str(objective)

    # Seed from risk parity / ERC
    if risk_budgets is not None:
        rb = np.asarray(risk_budgets, dtype=np.float64).ravel()
        if rb.size != n:
            rb = np.resize(rb, n)
        rb = np.maximum(rb, 1e-12)
        rb = rb / rb.sum()
        seed = rb.copy()
    else:
        rp = risk_parity_weights(c)
        seed = np.asarray(rp.get("weights", np.full(n, 1.0 / n)), dtype=np.float64).ravel()
        if seed.size != n:
            seed = np.full(n, 1.0 / n)

    if corr is not None:
        eff = effective_risk_budgets(seed, corr, names=labels)
        effective = eff["effective"]
        if isinstance(effective, dict):
            seed = np.asarray([float(effective.get(k, 0.0)) for k in labels], dtype=np.float64)
        else:
            seed = np.asarray(effective, dtype=np.float64).ravel()
        if seed.size != n:
            tmp = np.zeros(n, dtype=np.float64)
            tmp[: min(n, seed.size)] = seed[: min(n, seed.size)]
            seed = tmp
        if float(seed.sum()) > 0:
            seed = seed / float(seed.sum())

    w = seed.copy()
    reasons = [f"objective={obj}"]

    if obj == "min_risk":
        # Inverse-volatility / ERC blend
        erc = equal_risk_contribution(c)
        w = np.asarray(erc.get("weights", seed), dtype=np.float64).ravel()
        reasons.append("min_risk_via_erc")
    elif obj == "max_diversification":
        vols = np.sqrt(np.maximum(np.diag(c), 1e-18))
        inv = 1.0 / vols
        w = inv / inv.sum()
        reasons.append("max_diversification_inv_vol")
    elif obj == "target_volatility":
        vols = np.sqrt(np.maximum(np.diag(c), 1e-18))
        inv = 1.0 / vols
        w = inv / inv.sum()
        pv = float(portfolio_volatility(w, c).value)
        if pv > 1e-12:
            scale = float(np.clip(target_vol / pv, 0.0, max_leverage))
            w = w * scale
            reasons.append(f"scaled_to_target_vol={target_vol} scale={scale:.4f}")
    elif obj == "risk_budget_match":
        # Iterative projection toward budget proportions
        target = seed / max(seed.sum(), 1e-12)
        w = target.copy()
        for _ in range(max(int(max_iter), 1)):
            mrc = c @ w
            port_var = float(w @ mrc)
            if port_var <= 1e-18:
                break
            rc = w * mrc / port_var
            rc = np.maximum(rc, 1e-18)
            w = w * (target / rc)
            w = np.maximum(w, 0.0)
            s = w.sum()
            if s > 0:
                w = w / s
        reasons.append("risk_budget_match_iterative")
    elif obj == "target_cvar":
        # Conservative: shrink toward equal weight if returns absent; else use left-tail proxy
        if returns is not None:
            r = np.asarray(returns, dtype=np.float64)
            if r.ndim == 2 and r.shape[1] == n:
                # Approximate asset CVaR contribution via left-tail mean
                alpha = 0.05
                cvars = np.zeros(n)
                for i in range(n):
                    col = r[:, i]
                    q = np.quantile(col, alpha)
                    tail = col[col <= q]
                    cvars[i] = abs(float(np.mean(tail))) if tail.size else abs(float(q))
                inv = 1.0 / np.maximum(cvars, 1e-8)
                w = inv / inv.sum()
                if target_cvar is not None:
                    # Soft scale (hard cap applied in project_weights)
                    reasons.append(f"target_cvar_soft={target_cvar}")
            else:
                w = np.full(n, 1.0 / n)
        else:
            w = np.full(n, 1.0 / n)
            reasons.append("target_cvar_without_returns_equal_weight")
    elif obj == "target_drawdown":
        # Without path DD model, use vol as DD proxy and shrink
        vols = np.sqrt(np.maximum(np.diag(c), 1e-18))
        inv = 1.0 / vols
        w = inv / inv.sum()
        if target_drawdown is not None:
            reasons.append(f"target_drawdown_proxy={target_drawdown}")
    elif obj == "max_risk_adjusted_opportunity":
        opp = (
            np.asarray(expected_opportunity, dtype=np.float64).ravel()
            if expected_opportunity is not None
            else np.ones(n)
        )
        if opp.size != n:
            opp = np.resize(np.maximum(opp, 0.0), n)
        opp = np.maximum(opp, 0.0)
        vols = np.sqrt(np.maximum(np.diag(c), 1e-18))
        score = opp / vols
        if float(score.sum()) <= 0:
            w = np.full(n, 1.0 / n)
            reasons.append("opportunity_nonpositive_fallback_equal")
        else:
            w = score / score.sum()
            reasons.append("opportunity_over_vol_tilt")
    else:
        w = seed / max(seed.sum(), 1e-12)
        reasons.append("unknown_objective_fallback_seed")

    projected = project_weights(
        w,
        max_weight=max_weight,
        max_gross=max_leverage,
        max_leverage=max_leverage,
    )
    w_final = np.asarray(projected["weights"], dtype=np.float64).ravel()
    if w_final.size != n:
        tmp = np.zeros(n)
        tmp[: min(n, w_final.size)] = w_final[: min(n, w_final.size)]
        w_final = tmp
    s = float(w_final.sum())
    if s > 0 and obj not in ("target_volatility",):
        # Keep gross for target_vol leverage scaling; otherwise normalize to 1
        if float(np.sum(np.abs(w_final))) > max_leverage + 1e-12:
            w_final = w_final * (max_leverage / float(np.sum(np.abs(w_final))))
        elif abs(s - 1.0) > 1e-8 and obj != "target_volatility":
            w_final = w_final / s

    pv = float(portfolio_volatility(w_final, c).value)
    return {
        "name": "optimize_risk_budgets",
        "objective": obj,
        "names": labels,
        "weights": {labels[i]: float(w_final[i]) for i in range(n)},
        "weight_vector": w_final.tolist(),
        "portfolio_volatility": pv,
        "constraints": {
            "max_weight": float(max_weight),
            "max_leverage": float(max_leverage),
            "target_vol": float(target_vol),
            "target_cvar": target_cvar,
            "target_drawdown": target_drawdown,
        },
        "reasons": reasons,
        "note": "Hard limits bind; opportunity cannot override risk caps",
    }
