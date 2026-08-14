"""Standardized strategy scorecard."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.performance.returns import as_returns, cagr, total_return
from iqrp.app.backtesting.performance.risk_adjusted import (
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from iqrp.app.backtesting.performance.stability import stability_report
from iqrp.app.backtesting.performance.tail import conditional_value_at_risk
from iqrp.app.backtesting.performance.trade_metrics import turnover

__all__ = ["StrategyScorecard", "build_scorecard"]


@dataclass
class StrategyScorecard:
    """Institutional strategy scorecard — not return/Sharpe alone."""

    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    cvar: float = 0.0
    turnover: float = 0.0
    capacity: float | None = None
    transaction_costs: float = 0.0
    stability: float = 0.0
    regime_robustness: float | None = None
    out_of_sample: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize scorecard to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StrategyScorecard:
        """Construct from a mapping (unknown keys → metadata)."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known and k != "metadata"}
        meta = dict(data.get("metadata", {}))
        for k, v in data.items():
            if k not in known:
                meta[k] = v
        kwargs["metadata"] = meta
        return cls(**kwargs)

    def passes_gates(
        self,
        *,
        min_sharpe: float | None = None,
        max_drawdown: float | None = None,
        max_cvar: float | None = None,
        min_oos: float | None = None,
        min_stability: float | None = None,
        min_regime_robustness: float | None = None,
        max_turnover: float | None = None,
        max_costs: float | None = None,
        min_capacity: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate configurable promotion gates.

        A strategy is not promoted on return or Sharpe alone — gates cover
        risk, costs, stability, capacity, and OOS robustness.
        """
        checks: dict[str, bool] = {}
        if min_sharpe is not None:
            checks["sharpe"] = self.sharpe >= float(min_sharpe)
        if max_drawdown is not None:
            checks["max_drawdown"] = self.max_drawdown <= float(max_drawdown)
        if max_cvar is not None:
            checks["cvar"] = self.cvar <= float(max_cvar)
        if min_oos is not None:
            checks["out_of_sample"] = (
                self.out_of_sample is not None and self.out_of_sample >= float(min_oos)
            )
        if min_stability is not None:
            checks["stability"] = self.stability >= float(min_stability)
        if min_regime_robustness is not None:
            checks["regime_robustness"] = (
                self.regime_robustness is not None
                and self.regime_robustness >= float(min_regime_robustness)
            )
        if max_turnover is not None:
            checks["turnover"] = self.turnover <= float(max_turnover)
        if max_costs is not None:
            checks["transaction_costs"] = self.transaction_costs <= float(max_costs)
        if min_capacity is not None:
            checks["capacity"] = self.capacity is not None and self.capacity >= float(min_capacity)
        passed = all(checks.values()) if checks else True
        return {"passed": passed, "checks": checks}


def build_scorecard(
    returns: Any,
    *,
    positions: Any | None = None,
    costs: Any | None = None,
    capacity: float | None = None,
    regime_returns: Mapping[str, Any] | None = None,
    oos_returns: Any | None = None,
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
    stability_window: int = 63,
    confidence: float = 0.95,
    metadata: Mapping[str, Any] | None = None,
) -> StrategyScorecard:
    """Build a :class:`StrategyScorecard` from backtest outputs."""
    r = as_returns(returns)
    stab = stability_report(r, window=stability_window, periods_per_year=periods_per_year)
    sharpe_stats = stab["sharpe_stability"]
    # Stability score: higher when rolling Sharpe is positive and low-dispersion
    mean_s = float(sharpe_stats["mean"])
    std_s = float(sharpe_stats["std"])
    stability_score = float(mean_s / (1.0 + std_s)) if np.isfinite(mean_s) else 0.0

    to = float(turnover(positions)) if positions is not None else 0.0
    if costs is None:
        cost_total = 0.0
    else:
        c = as_returns(costs)
        cost_total = float(np.sum(np.abs(c)))

    regime_score: float | None = None
    if regime_returns:
        sharpes = []
        for series in regime_returns.values():
            sharpes.append(
                sharpe_ratio(series, risk_free=risk_free, periods_per_year=periods_per_year)
            )
        if sharpes:
            # Robustness: fraction of regimes with non-negative Sharpe, scaled by min
            arr = np.asarray(sharpes, dtype=np.float64)
            regime_score = float(np.mean(arr >= 0.0) * (1.0 + np.min(arr)))

    oos_metric: float | None = None
    if oos_returns is not None:
        oos_metric = sharpe_ratio(
            oos_returns, risk_free=risk_free, periods_per_year=periods_per_year
        )

    return StrategyScorecard(
        total_return=total_return(r),
        cagr=cagr(r, periods_per_year=periods_per_year),
        sharpe=sharpe_ratio(r, risk_free=risk_free, periods_per_year=periods_per_year),
        sortino=sortino_ratio(r, mar=risk_free, periods_per_year=periods_per_year),
        calmar=calmar_ratio(r, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(r),
        cvar=conditional_value_at_risk(r, confidence=confidence),
        turnover=to,
        capacity=None if capacity is None else float(capacity),
        transaction_costs=cost_total,
        stability=stability_score,
        regime_robustness=regime_score,
        out_of_sample=oos_metric,
        metadata=dict(metadata or {}),
    )
