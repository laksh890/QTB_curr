"""Cross-estimator disagreement for the Risk Intelligence Ensemble.

Pairs (when both present):
- historical VaR vs Monte Carlo VaR
- GARCH vol vs realized vol
- parametric ES vs historical ES
- normal correlation vs stress correlation
- model liquidity vs observed liquidity
"""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.ensemble.config import EnsembleSettings


def _extract_numeric(metrics: dict[str, Any], key: str) -> float | None:
    if key not in metrics:
        return None
    value = metrics[key]
    if isinstance(value, dict):
        for k in ("value", "score", "risk"):
            if k in value:
                try:
                    v = float(value[k])
                    return v if np.isfinite(v) else None
                except (TypeError, ValueError):
                    return None
        return None
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def relative_disagreement(a: float, b: float, *, eps: float = 1e-8) -> float:
    """Symmetric relative disagreement in [0, ∞), typically clipped later to [0, 1]."""
    denom = max(abs(a) + abs(b), eps)
    return float(abs(a - b) / denom)


def pair_disagreement(
    metrics: dict[str, Any],
    left: str,
    right: str,
    *,
    eps: float = 1e-8,
) -> dict[str, Any] | None:
    a = _extract_numeric(metrics, left)
    b = _extract_numeric(metrics, right)
    if a is None or b is None:
        return None
    rel = relative_disagreement(a, b, eps=eps)
    consensus = 0.5 * (a + b)
    # Uncertainty grows with disagreement and magnitude
    uncertainty = float(np.clip(rel * (1.0 + abs(consensus)), 0.0, 1.0))
    return {
        "pair": [left, right],
        "left": float(a),
        "right": float(b),
        "consensus": float(consensus),
        "disagreement": float(np.clip(rel, 0.0, 1.0)),
        "relative_disagreement": float(rel),
        "uncertainty": uncertainty,
        "available": True,
    }


# Convenience aliases so callers can supply either naming convention
_ALIAS_GROUPS: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = [
    (
        "hist_var_vs_mc_var",
        "var",
        ("var_historical", "historical_var", "var_hist"),
        ("var_monte_carlo", "monte_carlo_var", "var_mc"),
    ),
    (
        "garch_vs_realized_vol",
        "volatility",
        ("garch_vol", "garch_volatility", "conditional_vol"),
        ("realized_vol", "volatility", "vol"),
    ),
    (
        "parametric_vs_hist_es",
        "expected_shortfall",
        ("es_parametric", "parametric_es", "cvar_parametric"),
        ("es_historical", "historical_es", "cvar", "cvar_historical"),
    ),
    (
        "normal_vs_stress_corr",
        "correlation",
        ("corr_normal", "correlation_normal", "correlation"),
        ("corr_stress", "correlation_stress", "stress_correlation"),
    ),
    (
        "model_vs_observed_liq",
        "liquidity",
        ("liquidity_model", "model_liquidity"),
        ("liquidity_observed", "observed_liquidity", "liquidity_score"),
    ),
]


def compute_disagreement(
    metrics: dict[str, Any],
    *,
    settings: EnsembleSettings,
) -> dict[str, Any]:
    """Compute pairwise disagreement, consensus, and uncertainty summaries."""
    eps = float(settings.disagreement.relative_epsilon)
    pairs_out: list[dict[str, Any]] = []
    named: dict[str, Any] = {}

    # Configured explicit pairs
    for pair in settings.disagreement.pairs:
        if len(pair) != 2:
            continue
        left, right = str(pair[0]), str(pair[1])
        result = pair_disagreement(metrics, left, right, eps=eps)
        if result is not None:
            key = f"{left}_vs_{right}"
            result["name"] = key
            pairs_out.append(result)
            named[key] = result

    # Alias-group pairs (fill gaps when config keys absent but aliases present)
    for name, family, lefts, rights in _ALIAS_GROUPS:
        if name in named or any(name.endswith(k) for k in named):
            # already covered
            pass
        left_key = next((k for k in lefts if _extract_numeric(metrics, k) is not None), None)
        right_key = next((k for k in rights if _extract_numeric(metrics, k) is not None), None)
        if left_key is None or right_key is None or left_key == right_key:
            named[name] = {
                "name": name,
                "family": family,
                "available": False,
                "disagreement": None,
                "consensus": None,
                "uncertainty": None,
            }
            continue
        # Skip duplicate if already recorded for same keys
        already = any(
            p.get("pair") == [left_key, right_key] or p.get("pair") == [right_key, left_key]
            for p in pairs_out
        )
        if already:
            continue
        result = pair_disagreement(metrics, left_key, right_key, eps=eps)
        if result is None:
            continue
        result["name"] = name
        result["family"] = family
        pairs_out.append(result)
        named[name] = result

    available = [p for p in pairs_out if p.get("available")]
    if available:
        overall = float(np.mean([float(p["disagreement"]) for p in available]))
        uncertainty = float(np.mean([float(p["uncertainty"]) for p in available]))
        max_pair = max(available, key=lambda p: float(p["disagreement"]))
    else:
        overall = 0.0
        uncertainty = 0.0
        max_pair = None

    high = overall >= float(settings.disagreement.high_disagreement)
    return {
        "pairs": pairs_out,
        "by_name": named,
        "overall_disagreement": overall,
        "overall_uncertainty": uncertainty,
        "high_disagreement": high,
        "n_pairs_available": len(available),
        "max_disagreement_pair": max_pair,
        "consensus_note": (
            "Consensus is the midpoint of paired estimators; disagreement is relative |a-b|/(|a|+|b|)."
        ),
    }


class DisagreementAnalyzer:
    def __init__(self, settings: EnsembleSettings) -> None:
        self.settings = settings

    def analyze(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return compute_disagreement(metrics, settings=self.settings)
