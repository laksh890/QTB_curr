"""Research-score ranking across alpha candidates.

Ranking is triage only — Historical Sharpe alone cannot approve.
Statistical significance alone ≠ alpha.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return dict(item)
    if hasattr(item, "to_dict"):
        return dict(item.to_dict())
    return {"value": item}


def _extract_score(item: Mapping[str, Any]) -> float:
    """Composite research score for sorting (higher is better)."""
    # Explicit scores
    for key in ("research_score", "overall", "score"):
        if key in item and item[key] is not None:
            try:
                v = float(item[key])
                if np.isfinite(v):
                    return v
            except (TypeError, ValueError):
                pass
    score_obj = item.get("score")
    if isinstance(score_obj, Mapping) and "overall" in score_obj:
        try:
            v = float(score_obj["overall"])
            if np.isfinite(v):
                return v
        except (TypeError, ValueError):
            pass
    if hasattr(score_obj, "overall"):
        try:
            v = float(score_obj.overall)
            if np.isfinite(v):
                return v
        except (TypeError, ValueError):
            pass

    # Nested performance / report
    perf = item.get("performance")
    if hasattr(perf, "to_dict"):
        perf = perf.to_dict()
    if not isinstance(perf, Mapping):
        perf = item.get("report")
        if hasattr(perf, "to_dict"):
            perf = perf.to_dict()
        if isinstance(perf, Mapping):
            perf = perf.get("performance") or perf

    ic = float("nan")
    hit = float("nan")
    stab = float("nan")
    hyp = 0.0
    if isinstance(perf, Mapping):
        ic = float(perf.get("ic_mean", perf.get("ic", float("nan"))))
        hit = float(perf.get("hit_rate", float("nan")))
    if "ic" in item:
        try:
            ic = float(item["ic"])
        except (TypeError, ValueError):
            pass
    if "ic_mean" in item:
        try:
            ic = float(item["ic_mean"])
        except (TypeError, ValueError):
            pass
    if "stability" in item:
        try:
            stab = float(item["stability"])
        except (TypeError, ValueError):
            pass
    hyp_text = str(item.get("economic_hypothesis") or "")
    definition = item.get("definition")
    if not hyp_text and isinstance(definition, Mapping):
        hyp_text = str(definition.get("economic_hypothesis") or "")
    elif not hyp_text and hasattr(definition, "economic_hypothesis"):
        hyp_text = str(definition.economic_hypothesis or "")
    hyp = min(1.0, len(hyp_text.strip()) / 120.0) * 100.0

    pred = 0.0
    parts = 0
    if np.isfinite(ic):
        pred += min(100.0, abs(ic) / 0.1 * 100.0)
        parts += 1
    if np.isfinite(hit):
        pred += min(100.0, max(0.0, (hit - 0.5) / 0.2) * 100.0)
        parts += 1
    predictive = pred / max(parts, 1)
    stability = float(stab) if np.isfinite(stab) else 0.0
    # Do NOT use Sharpe in ranking gate — triage uses IC / hyp / stability
    return float(0.45 * predictive + 0.25 * stability + 0.30 * hyp)


def _name_of(item: Mapping[str, Any], idx: int) -> str:
    for key in ("name", "signal_name", "experiment_id", "definition_id"):
        if item.get(key):
            return str(item[key])
    definition = item.get("definition")
    if isinstance(definition, Mapping) and definition.get("name"):
        return str(definition["name"])
    if hasattr(item.get("definition"), "name"):
        return str(item["definition"].name)
    return f"candidate_{idx}"


def rank_candidates(
    candidates: Sequence[Any],
    *,
    descending: bool = True,
) -> list[dict[str, Any]]:
    """Rank candidates by composite research score (not approval).

    Returns a new list of dicts with ``rank``, ``research_score``, and
    original payload under ``candidate``. Sharpe is never used as a sole
    ranking / approval criterion.
    """
    ranked: list[dict[str, Any]] = []
    for i, raw in enumerate(candidates):
        d = _as_dict(raw)
        score = _extract_score(d)
        ranked.append(
            {
                "name": _name_of(d, i),
                "research_score": score,
                "candidate": d,
                "disclaimer": (
                    "Research triage ranking only. "
                    "Statistical significance alone ≠ alpha. "
                    "Historical Sharpe alone cannot approve."
                ),
            }
        )
    ranked.sort(key=lambda r: r["research_score"], reverse=descending)
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked
