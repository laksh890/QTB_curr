"""Performance attribution / decomposition."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from iqrp.app.backtesting.performance.returns import as_returns, total_return

__all__ = [
    "attribute_by_groups",
    "attribute_strategy",
    "attribute_signal",
    "attribute_asset",
    "attribute_sector",
    "attribute_factor",
    "attribute_market",
    "attribute_timeframe",
    "attribute_regime",
    "attribute_execution",
    "attribute_costs",
    "full_attribution",
]


def attribute_by_groups(
    contributions: Any,
    labels: Any,
) -> dict[str, float]:
    """Sum contribution series by group label.

    ``contributions`` may be (T,) with matching ``labels`` (T,), or (T, N)
    with ``labels`` length N (assets).
    """
    c = np.asarray(contributions, dtype=np.float64)
    labs = np.asarray(labels)
    out: dict[str, float] = {}
    if c.ndim == 1:
        if labs.size != c.size:
            raise ValueError("labels length must match contributions")
        for lab, v in zip(labs.tolist(), c.tolist()):
            key = str(lab)
            out[key] = out.get(key, 0.0) + float(v)
        return out
    if c.ndim == 2:
        if labs.size != c.shape[1]:
            raise ValueError("labels length must match contribution columns")
        totals = np.nansum(c, axis=0)
        for lab, v in zip(labs.tolist(), totals.tolist()):
            key = str(lab)
            out[key] = out.get(key, 0.0) + float(v)
        return out
    raise ValueError("contributions must be 1-D or 2-D")


def attribute_strategy(
    strategy_returns: Mapping[str, Any] | np.ndarray,
    labels: Any | None = None,
) -> dict[str, float]:
    """Decompose by strategy sleeve.

    Accepts a mapping ``{name: returns}`` or a 2-D array with ``labels``.
    """
    if isinstance(strategy_returns, Mapping):
        return {str(k): total_return(v) for k, v in strategy_returns.items()}
    arr = np.asarray(strategy_returns, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("strategy_returns array must be 2-D (T, S)")
    if labels is None:
        labels = [f"strategy_{i}" for i in range(arr.shape[1])]
    return {str(lab): total_return(arr[:, i]) for i, lab in enumerate(labels)}


def attribute_signal(
    signal_pnls: Mapping[str, Any] | np.ndarray,
    labels: Any | None = None,
) -> dict[str, float]:
    """Decompose by signal source."""
    return attribute_strategy(signal_pnls, labels=labels)


def attribute_asset(
    asset_returns: Any,
    weights: Any,
) -> dict[str, float]:
    """Asset contribution ≈ sum_t (w_{t,i} * r_{t,i})."""
    r = np.asarray(asset_returns, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("asset_returns must be (T, N)")
    if w.ndim == 1:
        contrib = w * np.nansum(r, axis=0)
        return {f"asset_{i}": float(contrib[i]) for i in range(contrib.size)}
    n = min(r.shape[0], w.shape[0])
    contrib = np.nansum(w[:n] * r[:n], axis=0)
    return {f"asset_{i}": float(contrib[i]) for i in range(contrib.size)}


def attribute_sector(
    asset_returns: Any,
    weights: Any,
    sectors: Any,
) -> dict[str, float]:
    """Roll asset contributions up to sectors."""
    asset = attribute_asset(asset_returns, weights)
    labs = np.asarray(sectors).reshape(-1)
    out: dict[str, float] = {}
    for i, lab in enumerate(labs.tolist()):
        key = str(lab)
        out[key] = out.get(key, 0.0) + asset.get(f"asset_{i}", 0.0)
    return out


def attribute_factor(
    factor_returns: Any,
    factor_exposures: Any,
) -> dict[str, float]:
    """Factor PnL ≈ sum_t (exposure_{t,k} * factor_return_{t,k})."""
    fr = np.asarray(factor_returns, dtype=np.float64)
    fe = np.asarray(factor_exposures, dtype=np.float64)
    if fr.ndim == 1:
        fr = fr.reshape(-1, 1)
    if fe.ndim == 1:
        fe = fe.reshape(-1, 1)
    n = min(fr.shape[0], fe.shape[0])
    k = min(fr.shape[1], fe.shape[1])
    contrib = np.nansum(fe[:n, :k] * fr[:n, :k], axis=0)
    return {f"factor_{i}": float(contrib[i]) for i in range(k)}


def attribute_market(
    returns: Any,
    market: Any,
    *,
    beta: float | None = None,
) -> dict[str, float]:
    """Split into market (beta * market) and residual alpha."""
    r = as_returns(returns)
    m = as_returns(market)
    n = min(r.size, m.size)
    r = r[:n]
    m = m[:n]
    if n < 2:
        return {"market": 0.0, "residual": total_return(r)}
    if beta is None:
        var_m = float(np.var(m, ddof=1))
        b = float(np.cov(r, m, ddof=1)[0, 1] / var_m) if var_m > 1e-18 else 0.0
    else:
        b = float(beta)
    market_leg = b * m
    resid = r - market_leg
    return {
        "market": total_return(market_leg),
        "residual": total_return(resid),
        "beta": b,
    }


def attribute_timeframe(
    returns: Any,
    timeframe_labels: Any,
) -> dict[str, float]:
    """Sum period returns by timeframe bucket label."""
    return attribute_by_groups(as_returns(returns), timeframe_labels)


def attribute_regime(
    returns: Any,
    regime_labels: Any,
) -> dict[str, float]:
    """Compounded return by regime label."""
    r = as_returns(returns)
    labs = np.asarray(regime_labels).reshape(-1)
    n = min(r.size, labs.size)
    out: dict[str, float] = {}
    for lab in sorted(set(str(x) for x in labs[:n].tolist())):
        mask = np.asarray([str(x) == lab for x in labs[:n].tolist()])
        out[lab] = total_return(r[:n][mask])
    return out


def attribute_execution(
    gross_returns: Any,
    net_returns: Any,
) -> dict[str, float]:
    """Execution drag = gross − net (total return space, additive approx)."""
    g = total_return(gross_returns)
    n = total_return(net_returns)
    return {"gross": g, "net": n, "execution_drag": float(g - n)}


def attribute_costs(
    returns: Any,
    *,
    commission: Any | None = None,
    spread: Any | None = None,
    slippage: Any | None = None,
    market_impact: Any | None = None,
    financing: Any | None = None,
    borrow: Any | None = None,
) -> dict[str, float]:
    """Decompose cost drag components (positive = cost)."""
    components = {
        "commission": commission,
        "spread": spread,
        "slippage": slippage,
        "market_impact": market_impact,
        "financing": financing,
        "borrow": borrow,
    }
    out: dict[str, float] = {"strategy_return": total_return(returns)}
    total_cost = 0.0
    for name, series in components.items():
        if series is None:
            out[name] = 0.0
            continue
        s = as_returns(series)
        # costs may be positive cost numbers; treat as drag magnitude
        val = float(np.sum(np.abs(s)))
        out[name] = val
        total_cost += val
    out["total_cost"] = total_cost
    return out


def full_attribution(
    *,
    returns: Any | None = None,
    strategy_returns: Any | None = None,
    signal_pnls: Any | None = None,
    asset_returns: Any | None = None,
    weights: Any | None = None,
    sectors: Any | None = None,
    factor_returns: Any | None = None,
    factor_exposures: Any | None = None,
    market: Any | None = None,
    timeframe_labels: Any | None = None,
    regime_labels: Any | None = None,
    gross_returns: Any | None = None,
    net_returns: Any | None = None,
    costs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all available attribution slices."""
    report: dict[str, Any] = {}
    if strategy_returns is not None:
        report["strategy"] = attribute_strategy(strategy_returns)
    if signal_pnls is not None:
        report["signal"] = attribute_signal(signal_pnls)
    if asset_returns is not None and weights is not None:
        report["asset"] = attribute_asset(asset_returns, weights)
        if sectors is not None:
            report["sector"] = attribute_sector(asset_returns, weights, sectors)
    if factor_returns is not None and factor_exposures is not None:
        report["factor"] = attribute_factor(factor_returns, factor_exposures)
    if returns is not None and market is not None:
        report["market"] = attribute_market(returns, market)
    if returns is not None and timeframe_labels is not None:
        report["timeframe"] = attribute_timeframe(returns, timeframe_labels)
    if returns is not None and regime_labels is not None:
        report["regime"] = attribute_regime(returns, regime_labels)
    if gross_returns is not None and net_returns is not None:
        report["execution"] = attribute_execution(gross_returns, net_returns)
    if costs is not None and returns is not None:
        report["costs"] = attribute_costs(returns, **dict(costs))
    return report
