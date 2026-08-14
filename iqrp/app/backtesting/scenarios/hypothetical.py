"""Hypothetical shock scenarios (price/vol/corr/liquidity/spread/cost/gap)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ShockKind = Literal[
    "price",
    "volatility",
    "correlation",
    "liquidity",
    "spread",
    "cost",
    "interest_rate",
    "fx",
    "gap",
]

__all__ = ["HypotheticalShock", "apply_hypothetical_shock", "run_hypothetical_scenario"]


@dataclass
class HypotheticalShock:
    """Specification for a single hypothetical market shock."""

    kind: ShockKind
    magnitude: float
    name: str = "shock"
    assets: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_2d_returns(returns: Any) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 1:
        return r.reshape(-1, 1)
    if r.ndim != 2:
        raise ValueError("returns must be 1-D or 2-D")
    return r


def apply_hypothetical_shock(
    returns: Any,
    shock: HypotheticalShock | dict[str, Any],
    *,
    cov: Any | None = None,
    spreads: Any | None = None,
    costs: Any | None = None,
    liquidity: Any | None = None,
) -> dict[str, Any]:
    """Apply a shock and return stressed returns plus auxiliary series."""
    if isinstance(shock, dict):
        shock = HypotheticalShock(
            kind=shock["kind"],  # type: ignore[arg-type]
            magnitude=float(shock.get("magnitude", 0.0)),
            name=str(shock.get("name", "shock")),
            assets=shock.get("assets"),
            metadata=dict(shock.get("metadata") or {}),
        )

    r = _as_2d_returns(returns)
    kind = str(shock.kind).lower()
    mag = float(shock.magnitude)
    stressed = r.copy()
    aux: dict[str, Any] = {}

    if kind == "price":
        # Additive return shock on the first period (instantaneous mark)
        stressed[0, :] = stressed[0, :] + mag
        aux["price_shock"] = mag
    elif kind == "volatility":
        # Scale residual deviations from mean by (1+mag)
        mu = np.mean(stressed, axis=0, keepdims=True)
        stressed = mu + (stressed - mu) * (1.0 + mag)
        aux["vol_scale"] = 1.0 + mag
    elif kind == "correlation":
        if cov is None:
            c = np.cov(r, rowvar=False) if r.shape[1] > 1 else np.array([[float(np.var(r))]])
        else:
            c = np.asarray(cov, dtype=np.float64)
        # Blend toward perfect correlation (mag>0) or independence (mag<0)
        vol = np.sqrt(np.clip(np.diag(c), 0.0, None))
        outer = np.outer(vol, vol)
        target = outer.copy()  # corr = 1
        indep = np.diag(np.diag(c))
        alpha = float(np.clip(mag, -1.0, 1.0))
        if alpha >= 0:
            stressed_cov = (1.0 - alpha) * c + alpha * target
        else:
            stressed_cov = (1.0 + alpha) * c + (-alpha) * indep
        aux["stressed_cov"] = stressed_cov
        # Resample standardized residuals through new corr structure (single draw path)
        # Keep historical path but report cov shock; returns unchanged except metadata
        aux["correlation_shift"] = alpha
    elif kind == "liquidity":
        base = (
            np.ones(r.shape[1])
            if liquidity is None
            else np.asarray(liquidity, dtype=np.float64).reshape(-1)
        )
        if base.size == 1:
            base = np.full(r.shape[1], float(base[0]))
        # Liquidity deterioration increases effective cost drag
        liq = np.clip(base * (1.0 - mag), 1e-8, None)
        drag = mag * 0.01 / liq  # simple inverse-liquidity cost
        stressed = stressed - drag.reshape(1, -1)
        aux["liquidity"] = liq
        aux["liquidity_drag"] = drag
    elif kind == "spread":
        base = (
            np.zeros(r.shape[1])
            if spreads is None
            else np.asarray(spreads, dtype=np.float64).reshape(-1)
        )
        if base.size == 1:
            base = np.full(r.shape[1], float(base[0]))
        spread = base + mag
        stressed = stressed - 0.5 * spread.reshape(1, -1) / max(r.shape[0], 1)
        aux["spreads"] = spread
    elif kind in ("cost", "interest_rate", "fx"):
        base = (
            np.zeros(r.shape[0])
            if costs is None
            else np.asarray(costs, dtype=np.float64).reshape(-1)
        )
        if base.size == 1:
            base = np.full(r.shape[0], float(base[0]))
        if base.size != r.shape[0]:
            base = np.resize(base, r.shape[0])
        stressed = stressed - (base.reshape(-1, 1) + mag)
        aux["cost_shock"] = mag
    elif kind == "gap":
        # Overnight gap: apply shock to a single bar (default first)
        idx = int(shock.metadata.get("gap_index", 0))
        idx = int(np.clip(idx, 0, r.shape[0] - 1))
        stressed[idx, :] = stressed[idx, :] + mag
        aux["gap_index"] = idx
        aux["gap_size"] = mag
    else:
        raise ValueError(f"unsupported shock kind: {shock.kind!r}")

    if stressed.shape[1] == 1:
        out_returns = stressed.reshape(-1)
    else:
        out_returns = stressed

    return {
        "name": shock.name,
        "kind": kind,
        "magnitude": mag,
        "returns": out_returns,
        "assets": shock.assets,
        "aux": aux,
        "metadata": dict(shock.metadata),
    }


def run_hypothetical_scenario(
    returns: Any,
    shocks: list[HypotheticalShock | dict[str, Any]],
    *,
    weights: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply a sequence of hypothetical shocks and summarize portfolio impact."""
    current = np.asarray(returns, dtype=np.float64)
    applied: list[dict[str, Any]] = []
    for sh in shocks:
        result = apply_hypothetical_shock(current, sh, **kwargs)
        current = result["returns"]
        applied.append({k: v for k, v in result.items() if k != "returns"})

    r = np.asarray(current, dtype=np.float64)
    if r.ndim == 2:
        n = r.shape[1]
        w = (
            np.full(n, 1.0 / max(n, 1))
            if weights is None
            else np.asarray(weights, dtype=np.float64).reshape(-1)
        )
        port = r @ w
    else:
        port = r.reshape(-1)

    baseline = np.asarray(returns, dtype=np.float64)
    if baseline.ndim == 2:
        n = baseline.shape[1]
        w = (
            np.full(n, 1.0 / max(n, 1))
            if weights is None
            else np.asarray(weights, dtype=np.float64).reshape(-1)
        )
        base_port = baseline @ w
    else:
        base_port = baseline.reshape(-1)

    pnl = float(np.sum(port - base_port[: port.size]))
    return {
        "name": "hypothetical",
        "kind": "hypothetical",
        "shocks": applied,
        "returns": port,
        "pnl": pnl,
        "loss": float(max(-pnl, 0.0)),
    }
