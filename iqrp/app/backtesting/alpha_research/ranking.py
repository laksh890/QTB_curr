"""Alpha Research Score and classification (configurable; not max-return)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from iqrp.app.backtesting.alpha_research.types import (
    DEFAULT_ALPHA_GATES,
    DEFAULT_ALPHA_SCORE_WEIGHTS,
    SAMPLE_TOO_SHORT_DISCLAIMER,
    AlphaClassification,
)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _norm_sharpe(s: float) -> float:
    return _clamp01((float(s) + 1.0) / 4.0)


def compute_alpha_research_score(
    metrics: Mapping[str, Any],
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """
    score = Σ w_i * component_i

    Components (higher better): net Sharpe, expectancy, OOS, IC, IC stability,
    drawdown (inverted), turnover (inverted), cost sensitivity, parameter/regime stability.
    Raw return is NOT the sole ranking criterion.
    """
    w = dict(DEFAULT_ALPHA_SCORE_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items()})
    s = sum(w.values()) or 1.0
    w = {k: v / s for k, v in w.items()}

    net_sharpe = float(metrics.get("net_sharpe", 0.0) or 0.0)
    expectancy = float(metrics.get("expectancy", metrics.get("net_alpha", 0.0)) or 0.0)
    oos = float(metrics.get("oos_sharpe", 0.0) or 0.0)
    ic = float(metrics.get("mean_ic", 0.0) or 0.0)
    ic_stab = float(metrics.get("ic_stability", 0.5) or 0.5)
    dd = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    turnover = abs(float(metrics.get("annualized_turnover", 0.0) or 0.0))
    cost_sens = 1.0 if metrics.get("alpha_survives_costs") else 0.2
    param_stab = float(metrics.get("parameter_stability", 0.5) or 0.5)
    regime_stab = float(metrics.get("regime_stability", 0.5) or 0.5)

    components = {
        "net_sharpe": _norm_sharpe(net_sharpe),
        "net_expectancy": _clamp01(0.5 + expectancy * 50.0),
        "oos": _norm_sharpe(oos),
        "ic": _clamp01(0.5 + ic * 2.0),
        "ic_stability": _clamp01(ic_stab),
        "drawdown": 1.0 - _clamp01(dd),
        "turnover": 1.0 - _clamp01(turnover / 50.0),
        "cost_sensitivity": float(cost_sens),
        "parameter_stability": _clamp01(param_stab),
        "regime_stability": _clamp01(regime_stab),
    }
    score = sum(w.get(k, 0.0) * components[k] for k in components)
    return {
        "score": float(score),
        "components": components,
        "weights": w,
        "formula_doc": compute_alpha_research_score.__doc__,
    }


def classify_alpha(
    metrics: Mapping[str, Any],
    *,
    gates: Mapping[str, Any] | None = None,
    n_sessions: int | None = None,
) -> tuple[AlphaClassification, str]:
    g = {**DEFAULT_ALPHA_GATES, **dict(gates or {})}
    if n_sessions is not None and n_sessions < int(g["min_sessions_for_significance"]):
        return AlphaClassification.SAMPLE_TOO_SHORT, SAMPLE_TOO_SHORT_DISCLAIMER

    if metrics.get("alpha_collapses_after_costs"):
        return AlphaClassification.COST_INEFFICIENT, "edge collapses after realistic costs"
    trades = int(metrics.get("trade_count", 0) or 0)
    if trades < int(g["min_trades"]):
        return AlphaClassification.INSUFFICIENT_DATA, f"trade_count {trades} < min_trades"
    if metrics.get("oos_evaluated") and float(metrics.get("oos_sharpe", 0.0) or 0.0) < float(
        g["min_oos_sharpe"]
    ):
        return AlphaClassification.OOS_FAILURE, "OOS Sharpe below threshold"
    if metrics.get("fragile"):
        return AlphaClassification.FRAGILE_ALPHA, "unstable across nearby parameters/horizons"
    dd = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    if dd > float(g["max_drawdown"]):
        return AlphaClassification.FRAGILE_ALPHA, "drawdown exceeds gate"

    net_sharpe = float(metrics.get("net_sharpe", 0.0) or 0.0)
    expectancy = float(metrics.get("expectancy", metrics.get("net_alpha", 0.0)) or 0.0)
    stab = float(metrics.get("parameter_stability", 1.0) or 1.0)
    survives = bool(metrics.get("alpha_survives_costs"))
    if (
        survives
        and net_sharpe > 0
        and expectancy >= float(g["min_net_expectancy"])
        and stab >= float(g["min_neighborhood_stability"])
        and (not metrics.get("oos_evaluated") or float(metrics.get("oos_sharpe", 0) or 0) >= float(g["min_oos_sharpe"]))
    ):
        # Still SAMPLE_TOO_SHORT if sessions insufficient — handled above
        return AlphaClassification.ROBUST_ALPHA, "passes configurable robust gates (research only)"
    if net_sharpe > 0 or expectancy > 0:
        return AlphaClassification.PROMISING_ALPHA, "positive research metrics but not fully robust"
    return AlphaClassification.FRAGILE_ALPHA, "does not meet promising/robust criteria"


__all__ = ["classify_alpha", "compute_alpha_research_score"]
