"""Frozen candidate verification — exact Prompt-42 / Prompt-39 definitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from iqrp.app.backtesting.final_holdout.protocol import DISCLAIMER, FROZEN_CANDIDATE_IDS

# Fields that define the trading strategy (must match P39 experiment + P42 carry-forward)
FREEZE_FIELDS = (
    "experiment_id",
    "kind",
    "source_id",
    "timeframe",
    "holding_bars",
    "direction",
    "family",
    "cost_scenario",
    "train_frac",
    "validation_frac",
    "purge_bars",
    "embargo_bars",
    "random_seed",
    "software_version",
)


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def definition_checksum(defn: dict[str, Any]) -> str:
    payload = {k: defn.get(k) for k in FREEZE_FIELDS}
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def load_p39_experiment(prompt39_dir: Path, experiment_id: str) -> dict[str, Any]:
    reg = json.loads((prompt39_dir / "experiment_registry.json").read_text(encoding="utf-8"))
    for e in reg["experiments"]:
        if e.get("experiment_id") == experiment_id:
            return e
    raise KeyError(experiment_id)


def load_p42_row(prompt42_dir: Path, experiment_id: str) -> dict[str, Any]:
    data = json.loads((prompt42_dir / "candidate_results.json").read_text(encoding="utf-8"))
    for r in data["results"]:
        if r.get("candidate_id") == experiment_id or r.get("experiment_id") == experiment_id:
            return r
    raise KeyError(experiment_id)


def freeze_candidates(
    *,
    prompt39_dir: str = "results/model_driven_alpha_campaign",
    prompt42_dir: str = "results/final_trading_validation",
    frozen_ids: tuple[str, ...] = FROZEN_CANDIDATE_IDS,
) -> dict[str, Any]:
    p39 = Path(prompt39_dir)
    p42 = Path(prompt42_dir)
    frozen = []
    all_ok = True
    for cid in frozen_ids:
        exp = load_p39_experiment(p39, cid)
        p42_row = load_p42_row(p42, cid)
        defn = {k: exp.get(k) for k in FREEZE_FIELDS}
        # Cross-check against P42 recorded fields
        mismatches = []
        for field, p42_key in (
            ("timeframe", "timeframe"),
            ("holding_bars", "holding_bars"),
            ("direction", "direction"),
            ("source_id", "signal_id"),
        ):
            a = defn.get(field)
            b = p42_row.get(p42_key)
            if str(a) != str(b):
                mismatches.append({"field": field, "p39": a, "p42": b})
        if p42_row.get("gate_status") != "PROFITABILITY_EVIDENCE":
            mismatches.append(
                {
                    "field": "gate_status",
                    "expected": "PROFITABILITY_EVIDENCE",
                    "p42": p42_row.get("gate_status"),
                }
            )
        checksum = definition_checksum(defn)
        ok = len(mismatches) == 0
        all_ok = all_ok and ok
        frozen.append(
            {
                "candidate_id": cid,
                "definition": defn,
                "lineage": exp.get("lineage"),
                "definition_checksum": checksum,
                "p42_reference": {
                    "oos_net_sharpe": p42_row.get("oos_net_sharpe"),
                    "oos_net_return": p42_row.get("oos_net_return"),
                    "oos_max_dd": p42_row.get("oos_max_dd"),
                    "trades_per_day": p42_row.get("trades_per_day"),
                    "gate_status": p42_row.get("gate_status"),
                    "model_family": p42_row.get("model_family"),
                    "behavior_class": p42_row.get("behavior_class"),
                },
                "cross_check_ok": ok,
                "mismatches": mismatches,
                "portfolio_treatment": "single-sleeve evaluation (no re-optimization)",
                "risk_constraints": "unchanged unified-pipeline defaults for recon smoke only",
                "cost_assumptions": "COST_SCENARIOS BASE/MODERATE/ADVERSE (unchanged)",
                "entry_exit_logic": (
                    "positions_from_signal(apply_direction_mask(raw_mtf_signal), holding_bars); "
                    "next-bar PnL via evaluate_cost_aware (pos[t-1]*ret[t])"
                ),
            }
        )
    aggregate = hashlib.sha256(
        "|".join(c["definition_checksum"] for c in frozen).encode()
    ).hexdigest()
    return {
        "disclaimer": DISCLAIMER,
        "n_frozen": len(frozen),
        "all_definitions_match": all_ok,
        "aggregate_definition_checksum": aggregate,
        "candidates": frozen,
        "status": "PASS" if all_ok else "FAIL",
    }


__all__ = ["freeze_candidates", "definition_checksum", "FREEZE_FIELDS"]
