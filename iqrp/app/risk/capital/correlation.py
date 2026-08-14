"""Strategy / factor / return / drawdown / tail correlation for capital sharing."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.market.correlation import correlation_matrix, ewma_correlation
from iqrp.app.risk.tail.drawdown import drawdown_series
from iqrp.app.risk.tail.tail_dependence import empirical_tail_dependence


def strategy_correlation(
    returns: Any,
    *,
    window: int | None = None,
    method: str = "pearson",
) -> dict[str, Any]:
    """Strategy return correlation matrix (point-in-time sample / EWMA)."""
    if str(method).lower() == "ewma":
        out = ewma_correlation(returns)
        out["name"] = "strategy_correlation"
        return out
    out = correlation_matrix(returns, window=window)
    out["name"] = "strategy_correlation"
    return out


def factor_correlation(
    factor_returns: Any,
    *,
    window: int | None = None,
) -> dict[str, Any]:
    """Factor return correlation matrix."""
    out = correlation_matrix(factor_returns, window=window)
    out["name"] = "factor_correlation"
    return out


def return_correlation(
    returns: Any,
    *,
    window: int | None = None,
) -> dict[str, Any]:
    """Asset/strategy return correlation (alias of market.correlation_matrix)."""
    out = correlation_matrix(returns, window=window)
    out["name"] = "return_correlation"
    return out


def drawdown_correlation(
    returns: Any,
    *,
    window: int | None = None,
) -> dict[str, Any]:
    """Correlation of drawdown series across strategies (co-underwater risk)."""
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2 or x.shape[1] == 0:
        return {
            "name": "drawdown_correlation",
            "matrix": [],
            "shape": [0, 0],
            "n_obs": 0,
            "method": "drawdown_pearson",
        }
    dd_cols = []
    for j in range(x.shape[1]):
        dd_cols.append(drawdown_series(x[:, j]))
    dd = np.column_stack(dd_cols)
    out = correlation_matrix(dd, window=window)
    out["name"] = "drawdown_correlation"
    out["method"] = "drawdown_pearson"
    return out


def tail_dependence_matrix(
    returns: Any,
    *,
    quantile: float = 0.05,
    tail: str = "lower",
) -> dict[str, Any]:
    """Pairwise empirical tail dependence matrix via risk.tail.tail_dependence."""
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[1] if x.ndim == 2 else 0
    if n == 0:
        return {
            "name": "tail_dependence_matrix",
            "matrix": [],
            "shape": [0, 0],
            "quantile": float(quantile),
            "tail": str(tail),
        }
    mat = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            m = empirical_tail_dependence(x[:, i], x[:, j], quantile=quantile, tail=tail)
            mat[i, j] = mat[j, i] = float(m.value)
    return {
        "name": "tail_dependence_matrix",
        "matrix": mat.tolist(),
        "shape": [n, n],
        "quantile": float(quantile),
        "tail": str(tail),
        "method": "empirical_tail_dependence",
    }


def correlation_crowding_scales(
    corr: Any,
    *,
    threshold: float = 0.60,
    floor: float = 0.25,
    names: list[str] | None = None,
) -> dict[str, float]:
    """Scale factors so correlated strategies share effective risk budget.

    For each name i, effective scale = 1 / (1 + sum_{j!=i} max(0, corr_ij - threshold)).
    Floor prevents total collapse; missing/invalid corr → identity (no upscale).
    """
    c = np.asarray(corr, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        n = len(names) if names else 0
        keys = names or [f"s{i}" for i in range(n)]
        return dict.fromkeys(keys, 1.0)
    n = c.shape[0]
    keys = names if names and len(names) == n else [f"s{i}" for i in range(n)]
    thr = float(threshold)
    fl = float(np.clip(floor, 0.0, 1.0))
    scales: dict[str, float] = {}
    for i in range(n):
        excess = 0.0
        for j in range(n):
            if i == j:
                continue
            cij = float(c[i, j]) if np.isfinite(c[i, j]) else 0.0
            excess += max(0.0, cij - thr)
        scale = 1.0 / (1.0 + excess)
        scales[keys[i]] = float(np.clip(scale, fl, 1.0))
    return scales


def effective_risk_budgets(
    risk_budgets: dict[str, float] | list[float] | np.ndarray,
    corr: Any,
    *,
    names: list[str] | None = None,
    threshold: float = 0.60,
    floor: float = 0.25,
) -> dict[str, Any]:
    """Apply correlation-aware scaling to nominal risk budgets."""
    if isinstance(risk_budgets, dict):
        keys = list(risk_budgets.keys()) if names is None else list(names)
        raw = np.asarray([float(risk_budgets.get(k, 0.0)) for k in keys], dtype=np.float64)
    else:
        raw = np.asarray(risk_budgets, dtype=np.float64).ravel()
        keys = names if names and len(names) == raw.size else [f"s{i}" for i in range(raw.size)]
    scales = correlation_crowding_scales(corr, threshold=threshold, floor=floor, names=keys)
    effective = {k: float(raw[i]) * scales[k] for i, k in enumerate(keys)}
    # Renormalize to preserve total budget mass when possible
    total_raw = float(np.sum(raw))
    total_eff = float(sum(effective.values()))
    if total_eff > 1e-12 and total_raw > 1e-12:
        factor = total_raw / total_eff
        effective = {k: v * factor for k, v in effective.items()}
    return {
        "name": "effective_risk_budgets",
        "nominal": {k: float(raw[i]) for i, k in enumerate(keys)},
        "scales": scales,
        "effective": effective,
        "threshold": float(threshold),
        "floor": float(floor),
    }
