"""Score market/tail/liquidity/concentration/correlation/drawdown/model/operational risk.

Scores are in [0, 1] with 1 = maximum risk. Dimension identity is preserved;
overall is a weighted synthesis, not a blind average of raw metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.ensemble.config import EnsembleSettings
from iqrp.app.risk.ensemble.types import RISK_DIMENSIONS, NormalizedMetric, RiskScore
from iqrp.app.risk.ensemble.weighting import resolve_weights


def _pick(
    normalized: dict[str, NormalizedMetric],
    keys: tuple[str, ...],
    *,
    default: float | None = None,
) -> float | None:
    vals: list[float] = []
    for k in keys:
        if k in normalized:
            vals.append(float(normalized[k].normalized_value))
    if not vals:
        return default
    return float(max(vals))


def _mean_present(vals: list[float | None]) -> float | None:
    present = [float(v) for v in vals if v is not None]
    if not present:
        return None
    return float(np.mean(present))


def score_dimensions(
    normalized: dict[str, NormalizedMetric],
    *,
    settings: EnsembleSettings,
    weights: dict[str, float] | None = None,
    missing_penalty: float = 0.85,
    disagreement: dict[str, Any] | None = None,
    regime: str = "normal",
) -> RiskScore:
    """Build per-dimension scores; missing dimensions use conservative penalty, never zero-risk."""

    market = _pick(normalized, ("volatility", "vol", "realized_vol", "garch_vol", "gap_risk"))
    tail = _pick(normalized, ("var", "cvar", "expected_shortfall", "es", "var_historical", "var_monte_carlo"))
    liquidity = _pick(normalized, ("liquidity_score", "liquidity", "liquidity_model", "liquidity_observed"))
    concentration = _pick(normalized, ("concentration", "hhi", "herfindahl"))
    correlation = _pick(normalized, ("correlation", "corr", "avg_correlation", "corr_normal", "corr_stress"))
    drawdown = _pick(normalized, ("drawdown", "current_drawdown", "max_drawdown", "dd"))
    model = _pick(normalized, ("model_risk", "model_disagreement", "forecast_uncertainty"))
    operational = _pick(normalized, ("operational", "ops_risk", "operational_risk"))

    # Disagreement elevates model risk only when paired estimators are available
    if model is None and disagreement and int(disagreement.get("n_pairs_available") or 0) > 0:
        overall_d = disagreement.get("overall_disagreement")
        if overall_d is not None:
            model = float(np.clip(float(overall_d), 0.0, 1.0))

    raw_dims: dict[str, float | None] = {
        "market": market,
        "tail": tail,
        "liquidity": liquidity,
        "concentration": concentration,
        "correlation": correlation,
        "drawdown": drawdown,
        "model": model,
        "operational": operational,
    }

    # Conservative fill for missing dimensions — do NOT treat as zero risk,
    # but do not let absent soft dimensions dominate the overall score.
    filled: dict[str, float] = {}
    missing_dims: list[str] = []
    observed = [v for v in raw_dims.values() if v is not None]
    if observed:
        conservative_floor = float(
            np.clip(0.5 * float(np.mean(observed)), 0.15, 0.55)
        )
    else:
        conservative_floor = float(np.clip(missing_penalty, 0.15, 0.85))
    for name, val in raw_dims.items():
        if val is None:
            filled[name] = float(np.clip(conservative_floor, 0.0, 1.0))
            missing_dims.append(name)
        else:
            filled[name] = float(np.clip(val, 0.0, 1.0))

    w = weights or resolve_weights(
        settings,
        dimension_scores=filled,
        disagreement=disagreement,
        regime=regime,
    )
    # Redistribute weight from missing dims onto observed dims (preserve identity in report)
    if missing_dims:
        present = [d for d in RISK_DIMENSIONS if d not in missing_dims]
        missing_w = sum(float(w.get(d, 0.0)) for d in missing_dims)
        present_w = sum(float(w.get(d, 0.0)) for d in present) or 1.0
        adj = dict(w)
        for d in missing_dims:
            adj[d] = float(w.get(d, 0.0)) * 0.25  # keep small residual identity contribution
        residual = missing_w - sum(adj[d] for d in missing_dims)
        for d in present:
            adj[d] = float(w.get(d, 0.0)) + residual * (float(w.get(d, 0.0)) / present_w)
        total = sum(adj.values()) or 1.0
        w = {d: adj[d] / total for d in RISK_DIMENSIONS}
    contributors = {d: float(w.get(d, 0.0) * filled[d]) for d in RISK_DIMENSIONS}
    overall = float(sum(contributors.values()))
    # Guard: overall should reflect elevated dimensions, not be diluted below max soft signal
    # when multiple critical dimensions are hot — use convex blend of weighted sum and max
    present_scores = [filled[d] for d in RISK_DIMENSIONS if d not in missing_dims]
    if present_scores:
        peak = float(max(present_scores))
        overall = float(np.clip(0.65 * overall + 0.35 * peak, 0.0, 1.0))

    return RiskScore(
        market=filled["market"],
        tail=filled["tail"],
        liquidity=filled["liquidity"],
        concentration=filled["concentration"],
        correlation=filled["correlation"],
        drawdown=filled["drawdown"],
        model=filled["model"],
        operational=filled["operational"],
        overall=float(np.clip(overall, 0.0, 1.0)),
        weights_applied=dict(w),
        contributors=contributors,
        metadata={
            "missing_dimensions": missing_dims,
            "missing_penalty_applied": conservative_floor,
            "regime": regime,
            "scoring_method": "identity_preserving_weighted_synthesis",
        },
    )


class RiskScorer:
    def __init__(self, settings: EnsembleSettings) -> None:
        self.settings = settings

    def score(
        self,
        normalized: dict[str, NormalizedMetric],
        **kwargs: Any,
    ) -> RiskScore:
        return score_dimensions(normalized, settings=self.settings, **kwargs)
