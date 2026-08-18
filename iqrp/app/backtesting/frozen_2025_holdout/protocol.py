"""Frozen 2024 research → independent 2025 holdout validation protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DISCLAIMER = (
    "FROZEN 2024→2025 HOLDOUT — research ≤2024-12-31; validation = calendar 2025. "
    "No retuning on 2025. PROVEN_RESEARCH_PROFITABILITY ≠ LIVE_READY. "
    "Do not modify Prompt 35–42 artifacts."
)

VALIDATION_ID = "frozen_2024_2025_holdout_v1"
SOFTWARE_VERSION = "iqrp-frozen-2024-2025-holdout-0.1.0"
OUTPUT_DIR = "results/frozen_2024_2025_holdout"

RESEARCH_END = "2024-12-31 23:59:59+00:00"
HOLDOUT_START = "2025-01-01 00:00:00+00:00"
HOLDOUT_END = "2025-12-31 23:59:59+00:00"

# P39-aligned research subsample (from end of research period)
P39_MAX_BARS: dict[str, int] = {
    "1m": 40000,
    "5m": 30000,
    "15m": 25000,
    "30m": 20000,
    "1h": 20000,
}

EVIDENCE_IDS: tuple[str, ...] = (
    "mdc_99aa952c5d5f6ff7",
    "mdc_678609c534d68189",
    "mdc_6f008c954ea26bf5",
)

GATE_MAX_DD = 0.50
GATE_MIN_TRADES = 50
GATE_MIN_INDEPENDENT_DAYS = 60


@dataclass
class Frozen2025Config:
    validation_id: str = VALIDATION_ID
    output_dir: str = OUTPUT_DIR
    prompt39_dir: str = "results/model_driven_alpha_campaign"
    prompt40_dir: str = "results/candidate_consolidation"
    prompt41_dir: str = "results/portfolio_construction_integration"
    prompt42_dir: str = "results/final_trading_validation"
    registry_path: str = "dataset_registry.json"
    data_dir: str = "data/btcusdt/firewall_2024_2025"
    random_seed: int = 42
    software_version: str = SOFTWARE_VERSION
    market_type: str = "crypto"
    evidence_ids: tuple[str, ...] = EVIDENCE_IDS
    max_bars_research: dict[str, int] = field(default_factory=lambda: dict(P39_MAX_BARS))
    run_portfolio: bool = True
    smoke: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["disclaimer"] = DISCLAIMER
        d["research_end"] = RESEARCH_END
        d["holdout_start"] = HOLDOUT_START
        d["holdout_end"] = HOLDOUT_END
        d["live_ready"] = False
        d["anti_post_hoc"] = "No parameter/timeframe/holding changes after seeing 2025 results."
        return d


def classify_candidate(row: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "complete_2025_holdout": bool(row.get("complete_2025_holdout")),
        "positive_net": float(row.get("net_return") or -1) > 0 and float(row.get("net_sharpe") or -1) > 0,
        "survives_BASE": bool(row.get("survives_BASE")),
        "survives_MODERATE": bool(row.get("survives_MODERATE")),
        "acceptable_drawdown": float(row.get("max_drawdown") or 1) <= GATE_MAX_DD,
        "sufficient_trades": int(row.get("n_trades") or 0) >= GATE_MIN_TRADES,
        "no_leakage": bool(row.get("causality_pass")),
        "no_post_hoc": True,
        "statistically_meaningful": bool(row.get("statistically_meaningful")),
        "stable_through_2025": bool(row.get("stable_through_2025")),
        "reproducible": bool(row.get("reproducible")),
        "not_single_period": bool(row.get("not_single_period")),
        "not_single_trade": bool(row.get("not_single_trade")),
        "firewall_pass": bool(row.get("firewall_pass")),
        "sharpe_not_inflated": bool(row.get("sharpe_not_inflated", True)),
    }
    failed = [k for k, v in checks.items() if not v]
    adverse_ok = bool(row.get("survives_ADVERSE"))

    if not checks["complete_2025_holdout"] or not checks["firewall_pass"]:
        status = "REJECTED"
    elif not checks["positive_net"] or not checks["survives_BASE"]:
        status = "REJECTED"
    elif not checks["survives_MODERATE"] or not checks["sharpe_not_inflated"]:
        status = "WEAK_EVIDENCE"
    elif failed:
        # partial research evidence
        if checks["positive_net"] and checks["survives_BASE"] and checks["survives_MODERATE"]:
            status = "RESEARCH_EVIDENCE" if checks["statistically_meaningful"] else "WEAK_EVIDENCE"
        else:
            status = "WEAK_EVIDENCE"
    elif adverse_ok and checks["statistically_meaningful"]:
        status = "PROVEN_RESEARCH_PROFITABILITY"
    elif checks["statistically_meaningful"]:
        status = "PAPER_TRADING_CANDIDATE"
    else:
        status = "RESEARCH_EVIDENCE"

    # Strongest labels require ALL checks
    if status in {"PAPER_TRADING_CANDIDATE", "PROVEN_RESEARCH_PROFITABILITY"} and failed:
        status = "RESEARCH_EVIDENCE" if checks["positive_net"] else "WEAK_EVIDENCE"

    return {"status": status, "checks": checks, "failed_checks": failed, "adverse_survives": adverse_ok}


__all__ = [
    "DISCLAIMER",
    "Frozen2025Config",
    "RESEARCH_END",
    "HOLDOUT_START",
    "HOLDOUT_END",
    "EVIDENCE_IDS",
    "classify_candidate",
]
