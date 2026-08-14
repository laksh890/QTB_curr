"""Horizon research reports and machine-readable matrix."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from iqrp.app.backtesting.horizon.config import HorizonResearchConfig
from iqrp.app.backtesting.horizon.types import HorizonResult, HorizonStatus, Timeframe


def build_horizon_matrix(results: Sequence[HorizonResult]) -> list[dict[str, Any]]:
    """strategy × instrument × data × signal × holding matrix rows."""
    rows: list[dict[str, Any]] = []
    for r in results:
        m = r.metrics
        tfreq = r.trade_frequency
        rows.append(
            {
                "strategy": r.spec.strategy_id,
                "instrument": r.spec.instrument,
                "data_timeframe": str(r.spec.data_timeframe),
                "signal_timeframe": str(r.spec.signal_timeframe),
                "holding": str(r.spec.holding),
                "holding_bars": r.spec.holding.bars,
                "trades": int(m.get("trade_count", tfreq.get("total_trades", 0)) or 0),
                "trades_per_day": float(tfreq.get("trades_per_day", 0.0) or 0.0),
                "long_trades": int(tfreq.get("long_trades", m.get("long_trade_count", 0)) or 0),
                "short_trades": int(tfreq.get("short_trades", m.get("short_trade_count", 0)) or 0),
                "gross_return": m.get("total_return_gross"),
                "net_return": m.get("total_return_net"),
                "gross_sharpe": m.get("gross_sharpe"),
                "net_sharpe": m.get("net_sharpe"),
                "sortino": m.get("sortino"),
                "max_dd": m.get("maximum_drawdown"),
                "turnover": (r.turnover or {}).get("annualized_turnover"),
                "transaction_costs": (r.costs or {}).get("transaction_costs"),
                "oos_net_sharpe": (r.oos or {}).get("net_sharpe"),
                "robustness_score": r.robustness_score,
                "status": r.status.value,
                "reason": r.reason,
                "result_class": "research_simulated",
                "disclaimer": r.disclaimer,
            }
        )
    return rows


def build_horizon_report(
    results: Sequence[HorizonResult],
    *,
    config: HorizonResearchConfig,
    native: Timeframe,
    availability: dict[str, Any],
    selection: dict[str, Any],
    multiple_testing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full horizon research report payload."""
    matrix = build_horizon_matrix(results)
    detailed = []
    for r in results:
        d = r.to_dict()
        # strip non-serializable helpers
        d["metrics"] = {
            k: v for k, v in d["metrics"].items() if not str(k).startswith("_")
        }
        detailed.append(d)

    return {
        "title": "Horizon Research Report",
        "dataset": {
            "native_frequency": str(native),
            "availability": availability,
            "instrument": config.instrument,
        },
        "strategy": config.strategy_id,
        "config": config.to_dict(),
        "matrix": matrix,
        "results": detailed,
        "selection": selection,
        "multiple_testing": dict(multiple_testing or {}),
        "classifications_present": sorted({r.status.value for r in results}),
        "n_unavailable": sum(1 for r in results if r.status == HorizonStatus.UNAVAILABLE),
        "n_robust": sum(1 for r in results if r.status == HorizonStatus.ROBUST),
        "disclaimers": [
            "Research / simulated results only — not live performance.",
            "Capacity estimates are ESTIMATED / MODEL-BASED.",
            "Positive backtest ≠ profitable live trading.",
            "Intraday horizons require real intraday data; finer-than-native "
            "frequencies are UNAVAILABLE and are never fabricated.",
            "BEST ROBUST HORIZON is gated and may differ from best in-sample.",
        ],
    }


__all__ = ["build_horizon_matrix", "build_horizon_report"]
