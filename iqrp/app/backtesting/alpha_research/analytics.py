"""Alpha analytics: IC, decay, TOD, costs, turnover, correlation, incremental, importance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.types import DEFAULT_FORWARD_HORIZONS
from iqrp.app.backtesting.horizon.costs import apply_cost_drag, gross_vs_net_sharpe
from iqrp.app.backtesting.horizon.half_life import signal_half_life_report
from iqrp.app.backtesting.horizon.trade_analytics import trade_frequency_report
from iqrp.app.backtesting.horizon.turnover import turnover_report
from iqrp.app.backtesting.performance.returns import as_returns, total_return
from iqrp.app.backtesting.performance.risk_adjusted import sharpe_ratio
from iqrp.app.backtesting.performance.trade_metrics import trades_from_positions


def forward_returns_matrix(prices: Any, horizons: Sequence[int]) -> dict[int, np.ndarray]:
    px = np.asarray(prices, dtype=np.float64).reshape(-1)
    out: dict[int, np.ndarray] = {}
    for h in horizons:
        h = int(h)
        fr = np.full(px.size, np.nan)
        if h > 0 and px.size > h:
            fr[: px.size - h] = px[h:] / px[: px.size - h] - 1.0
        out[h] = fr
    return out


def timeseries_ic_report(
    signal: Any,
    prices: Any,
    *,
    horizons: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Time-series predictive correlation (NOT cross-sectional IC)."""
    hs = list(horizons or DEFAULT_FORWARD_HORIZONS)
    sig = np.asarray(signal, dtype=np.float64).reshape(-1)
    frs = forward_returns_matrix(prices, hs)
    by_h: dict[str, Any] = {}
    ics_p, ics_s = [], []
    for h in hs:
        fr = frs[h]
        n = min(sig.size, fr.size)
        mask = np.isfinite(sig[:n]) & np.isfinite(fr[:n]) & (np.abs(sig[:n]) > 0)
        if mask.sum() < 5:
            by_h[str(h)] = {"pearson_ic": None, "spearman_ic": None, "n": int(mask.sum())}
            continue
        x, y = sig[:n][mask], fr[:n][mask]
        pear = float(np.corrcoef(x, y)[0, 1])
        # spearman via rank
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        spear = float(np.corrcoef(rx, ry)[0, 1])
        ics_p.append(pear)
        ics_s.append(spear)
        by_h[str(h)] = {
            "pearson_ic": pear,
            "spearman_ic": spear,
            "n": int(mask.sum()),
            "metric_type": "time_series_predictive_correlation",
        }
    arr_p = np.asarray(ics_p, dtype=np.float64)
    return {
        "metric_type": "time_series_predictive_correlation",
        "not_cross_sectional_ic": True,
        "by_horizon": by_h,
        "mean_ic": float(np.mean(arr_p)) if arr_p.size else None,
        "ic_std": float(np.std(arr_p, ddof=1)) if arr_p.size > 1 else None,
        "ic_ir": float(np.mean(arr_p) / (np.std(arr_p, ddof=1) + 1e-12)) if arr_p.size > 1 else None,
        "positive_ic_pct": float(np.mean(arr_p > 0)) if arr_p.size else None,
        "disclaimer": "Single-instrument IC is time-series correlation, not cross-sectional IC.",
    }


def decay_curve(signal: Any, prices: Any, horizons: Sequence[int] | None = None) -> dict[str, Any]:
    report = signal_half_life_report(signal, prices, horizons=horizons or DEFAULT_FORWARD_HORIZONS)
    by = report.get("by_horizon") or {}
    means = {int(k): v.get("mean_forward_return") for k, v in by.items() if v.get("mean_forward_return") is not None}
    peak = max(means, key=lambda k: abs(means[k] or 0.0)) if means else None
    # sign reversal: first horizon where mean flips vs peak sign
    reversal = None
    if peak is not None and means.get(peak) is not None:
        sign = np.sign(means[peak])
        for h in sorted(means):
            if h > peak and means[h] is not None and np.sign(means[h]) == -sign and abs(means[h]) > 1e-12:
                reversal = h
                break
    return {
        **report,
        "peak_predictive_horizon_bars": peak,
        "sign_reversal_horizon_bars": reversal,
        "note": "Do not select peak purely in-sample as the trading horizon.",
    }


def time_of_day_report(
    timestamps: Any,
    signal: Any,
    returns: Any,
    *,
    timezone: str = "Asia/Kolkata",
) -> dict[str, Any]:
    ts = pd.to_datetime(pd.Series(timestamps), utc=True)
    # NSE equity research defaults to Asia/Kolkata; crypto campaigns pass UTC.
    local = ts.dt.tz_convert(timezone)
    sig = np.asarray(signal, dtype=np.float64)
    ret = np.asarray(returns, dtype=np.float64)
    df = pd.DataFrame(
        {
            "hour": local.dt.hour,
            "minute_bucket": (local.dt.hour * 60 + local.dt.minute) // 30 * 30,
            "signal": sig,
            "ret": ret,
        }
    )
    df = df[np.isfinite(df["ret"])]

    def _bucket(col: str) -> dict[str, Any]:
        g = df.groupby(col, sort=True)
        out = {}
        for k, part in g:
            r = part["ret"].to_numpy()
            s = part["signal"].to_numpy()
            out[str(k)] = {
                "n": int(len(part)),
                "avg_return": float(np.mean(r)) if len(r) else None,
                "volatility": float(np.std(r, ddof=1)) if len(r) > 1 else None,
                "trade_freq": float(np.mean(np.abs(np.diff(np.sign(s), prepend=0)) > 0)) if len(s) else 0.0,
                "mean_abs_signal": float(np.nanmean(np.abs(s))) if len(s) else None,
            }
        return out

    return {
        "by_hour": _bucket("hour"),
        "by_minute_bucket": _bucket("minute_bucket"),
        "note": "Diagnostics for open/midday/close effects — do not auto-trade these.",
        "timezone": str(timezone),
    }


def positions_from_signal(signal: pd.Series, holding_bars: int) -> pd.Series:
    from iqrp.app.backtesting.alpha_research.signals import apply_holding

    return apply_holding(signal.fillna(0.0), holding_bars)


def evaluate_cost_aware(
    positions: pd.Series,
    bar_returns: pd.Series,
    *,
    commission_bps: float = 1.0,
    spread_bps: float = 2.0,
    slippage_bps: float = 2.0,
    periods_per_year: float = 252.0 * 75.0,
    timestamps: Any | None = None,
    n_calendar_days: int | None = None,
) -> dict[str, Any]:
    pos = positions.to_numpy(dtype=np.float64)
    rets = bar_returns.to_numpy(dtype=np.float64)
    gross = np.zeros_like(rets)
    gross[1:] = pos[:-1] * rets[1:]  # no lookahead
    turnover = np.abs(np.diff(pos, prepend=0.0))
    cost = apply_cost_drag(
        gross,
        commission_bps=commission_bps,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        turnover_per_period=turnover,
    )
    gn = gross_vs_net_sharpe(cost["gross_returns"], cost["net_returns"], periods_per_year=periods_per_year)
    survives = bool(cost["net_pnl"] > 0 and gn["net_sharpe"] > 0 and not gn.get("cost_inefficient"))
    trades = trades_from_positions(pos, rets)
    # enrich sides
    enriched = []
    for t in trades:
        enriched.append(
            {
                **t,
                "side": "LONG" if t.get("direction", 0) > 0 else "SHORT",
                "holding": t.get("holding"),
                "pnl": float(t["pnl"]) if np.isfinite(t.get("pnl", np.nan)) else 0.0,
            }
        )
    freq = trade_frequency_report(enriched, timestamps=timestamps)
    # When trade rows lack timestamps, fall back to calendar span
    if n_calendar_days is not None and int(n_calendar_days) > 0:
        if int(freq.get("n_trading_days_with_trades") or 0) <= 1:
            n_tr = int(freq.get("total_trades") or 0)
            freq["trades_per_day"] = float(n_tr / float(n_calendar_days))
            freq["calendar_days"] = int(n_calendar_days)
            freq["trades_per_day_method"] = "total_trades / n_calendar_days"
    to = turnover_report(pos, periods_per_day=max(periods_per_year / 252.0, 1.0), net_pnl=cost["net_pnl"], net_alpha=cost["net_alpha"])
    long_tr = [t for t in enriched if t.get("side") == "LONG"]
    short_tr = [t for t in enriched if t.get("side") == "SHORT"]
    sig_side = np.sign(pos)
    side_counts = {
        "long_observations": int(np.sum(sig_side > 0)),
        "short_observations": int(np.sum(sig_side < 0)),
        "flat_observations": int(np.sum(sig_side == 0)),
        "long_trades": int(len(long_tr)),
        "short_trades": int(len(short_tr)),
        "long_pnl": float(np.sum([t["pnl"] for t in long_tr])) if long_tr else 0.0,
        "short_pnl": float(np.sum([t["pnl"] for t in short_tr])) if short_tr else 0.0,
    }
    edge_trade = float(np.mean([t["pnl"] for t in enriched])) if enriched else 0.0
    return {
        "gross_alpha": cost["gross_alpha"],
        "net_alpha": cost["net_alpha"],
        "gross_pnl": cost["gross_pnl"],
        "net_pnl": cost["net_pnl"],
        "commissions": cost["commission"],
        "spread_cost": cost["spread_cost"],
        "slippage": cost["slippage"],
        "transaction_costs": cost["transaction_costs"],
        "gross_sharpe": gn["gross_sharpe"],
        "net_sharpe": gn["net_sharpe"],
        "gross_edge_per_trade": float(np.mean([t["pnl"] for t in enriched])) if enriched else 0.0,
        "net_edge_per_trade": edge_trade,
        "cost_per_trade": float(cost["transaction_costs"] / max(len(enriched), 1)),
        "cost_as_pct_of_gross": float(
            abs(cost["transaction_costs"]) / max(abs(cost["gross_pnl"]), 1e-12)
        ),
        "alpha_survives_costs": survives,
        "alpha_collapses_after_costs": bool(gn.get("cost_inefficient") or (cost["gross_pnl"] > 0 >= cost["net_pnl"])),
        "turnover": to,
        "trade_frequency": freq,
        "side_counts": side_counts,
        "trades": enriched,
        "gross_returns": cost["gross_returns"],
        "net_returns": cost["net_returns"],
        "positions": pos,
    }


def signal_correlation_matrix(signals: Mapping[str, Any]) -> dict[str, Any]:
    names = list(signals.keys())
    cols = [np.asarray(signals[n], dtype=np.float64).reshape(-1) for n in names]
    n = min(c.size for c in cols) if cols else 0
    mat = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            mask = np.isfinite(a[:n]) & np.isfinite(b[:n])
            if mask.sum() >= 5:
                mat[i, j] = float(np.corrcoef(a[:n][mask], b[:n][mask])[0, 1])
    return {
        "signals": names,
        "correlation": mat.tolist(),
        "note": "Highly correlated signals are not independent discoveries.",
    }


def return_correlation_matrix(returns_map: Mapping[str, Any]) -> dict[str, Any]:
    return signal_correlation_matrix(returns_map)


def incremental_alpha(
    base_eval: Mapping[str, Any],
    combined_eval: Mapping[str, Any],
) -> dict[str, Any]:
    def _g(d, k, default=0.0):
        return float(d.get(k, default) or default)

    return {
        "marginal_net_return": _g(combined_eval, "net_pnl") - _g(base_eval, "net_pnl"),
        "marginal_sharpe": _g(combined_eval, "net_sharpe") - _g(base_eval, "net_sharpe"),
        "marginal_turnover": _g(combined_eval.get("turnover") or {}, "annualized_turnover")
        - _g(base_eval.get("turnover") or {}, "annualized_turnover"),
        "marginal_drawdown": None,  # filled by caller if available
        "note": "Incremental alpha is a research diagnostic, not proof of independent edge.",
    }


def permutation_importance(
    signal: np.ndarray,
    forward: np.ndarray,
    *,
    n_perm: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Permute signal; importance = baseline |IC| − mean permuted |IC|. Not causality."""
    rng = np.random.default_rng(seed)
    s = np.asarray(signal, dtype=np.float64)
    f = np.asarray(forward, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(f)
    if mask.sum() < 10:
        return {"importance": None, "n": int(mask.sum()), "note": "insufficient data"}
    x, y = s[mask], f[mask]
    base = abs(float(np.corrcoef(x, y)[0, 1]))
    nulls = []
    for _ in range(n_perm):
        xp = rng.permutation(x)
        nulls.append(abs(float(np.corrcoef(xp, y)[0, 1])))
    imp = base - float(np.mean(nulls))
    return {
        "baseline_abs_ic": base,
        "null_mean_abs_ic": float(np.mean(nulls)),
        "importance": imp,
        "n_perm": n_perm,
        "disclaimer": "Feature importance ≠ causality.",
    }


def parameter_stability(
    scores: Mapping[str, float],
    *,
    center_key: str,
    fragile_gap: float = 0.5,
) -> dict[str, Any]:
    """Flag fragile if center score spikes vs neighbors."""
    if center_key not in scores:
        return {"fragile": True, "reason": "center missing"}
    center = float(scores[center_key])
    neighbors = [float(v) for k, v in scores.items() if k != center_key]
    if not neighbors:
        return {"fragile": True, "stability_score": 0.0, "reason": "no neighbors"}
    mean_n = float(np.mean(neighbors))
    std = float(np.std(list(scores.values()), ddof=1)) if len(scores) > 1 else 0.0
    mean = float(np.mean(list(scores.values())))
    cv = abs(std / mean) if abs(mean) > 1e-12 else 1.0
    stability = float(max(0.0, min(1.0, 1.0 - cv)))
    fragile = bool(center > 0 and (mean_n <= 0 or (center - mean_n) > fragile_gap * max(abs(center), 1e-9)))
    return {
        "stability_score": stability,
        "fragile": fragile,
        "center": center,
        "neighbor_mean": mean_n,
        "reason": "FRAGILE parameter island" if fragile else "stable across nearby parameters",
    }


__all__ = [
    "decay_curve",
    "evaluate_cost_aware",
    "forward_returns_matrix",
    "incremental_alpha",
    "parameter_stability",
    "permutation_importance",
    "positions_from_signal",
    "return_correlation_matrix",
    "signal_correlation_matrix",
    "time_of_day_report",
    "timeseries_ic_report",
]
