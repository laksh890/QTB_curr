"""Independent final holdout validation protocol (frozen gates).

Do not retune candidates. Do not manufacture independence.
LIVE_READY remains FALSE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DISCLAIMER = (
    "FINAL HOLDOUT VALIDATION — independent chronological holdout of frozen Prompt-42 alphas. "
    "HOLDOUT REPLICATED ≠ PROVEN PROFITABLE ≠ PAPER READY ≠ LIVE READY. "
    "Do not modify P35–P42 artifacts. Do not retune after observing holdout results."
)

VALIDATION_ID = "final_holdout_validation_v1"
SOFTWARE_VERSION = "iqrp-final-holdout-0.1.0"
RANDOM_SEED = 42
OUTPUT_DIR = "results/final_holdout_validation"

FROZEN_CANDIDATE_IDS: tuple[str, ...] = (
    "mdc_99aa952c5d5f6ff7",
    "mdc_6f008c954ea26bf5",
    "mdc_678609c534d68189",
)

PROMPT42_DIR = "results/final_trading_validation"
PROMPT39_DIR = "results/model_driven_alpha_campaign"
PROMPT40_DIR = "results/candidate_consolidation"

# Paper-trading gate (ALL required). Not a single Sharpe threshold.
PAPER_GATE_REQUIRED = (
    "untouched_holdout_exists",
    "candidate_definition_checksum_match",
    "causality_pass",
    "data_provenance_pass",
    "positive_net_performance",
    "survives_BASE",
    "survives_MODERATE",
    "acceptable_drawdown",
    "not_regime_only",
    "reproducible",
    "reconciliation_pass",
    "no_high_severity_defect",
    "holdout_sample_adequate",
)

GATE_MAX_DD = 0.50
GATE_MIN_HOLDOUT_CALENDAR_DAYS = 7
GATE_MIN_HOLDOUT_TRADES = 30
GATE_MIN_N_EFF = 50.0
GATE_MODERATE_COLLAPSE_SHARPE = -1.0  # below → collapse under MODERATE

DEGRADATION_STABLE_MAX = 0.35  # relative Sharpe drop
DEGRADATION_MODERATE_MAX = 0.70


@dataclass
class FinalHoldoutConfig:
    validation_id: str = VALIDATION_ID
    output_dir: str = OUTPUT_DIR
    prompt42_dir: str = PROMPT42_DIR
    prompt39_dir: str = PROMPT39_DIR
    prompt40_dir: str = PROMPT40_DIR
    registry_path: str = "dataset_registry.json"
    frozen_ids: tuple[str, ...] = FROZEN_CANDIDATE_IDS
    random_seed: int = RANDOM_SEED
    software_version: str = SOFTWARE_VERSION
    market_type: str = "crypto"
    warmup_bars: dict[str, int] = field(
        default_factory=lambda: {"1m": 10000, "5m": 5000, "15m": 3000, "30m": 2000, "1h": 1500}
    )
    smoke: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["disclaimer"] = DISCLAIMER
        d["paper_gate_required"] = list(PAPER_GATE_REQUIRED)
        d["live_ready"] = False
        d["anti_post_hoc"] = (
            "After holdout results: do not change TF/holding/thresholds/models/costs/candidates."
        )
        return d


def classify_degradation(p42_sharpe: float | None, holdout_sharpe: float | None) -> str:
    if p42_sharpe is None or holdout_sharpe is None:
        return "FAILED_REPLICATION"
    if not (p42_sharpe == p42_sharpe) or not (holdout_sharpe == holdout_sharpe):  # NaN
        return "FAILED_REPLICATION"
    if holdout_sharpe <= 0 and p42_sharpe > 0:
        return "FAILED_REPLICATION"
    if p42_sharpe <= 0:
        # research was already weak; judge holdout absolute
        return "STABLE" if holdout_sharpe > 0 else "FAILED_REPLICATION"
    drop = (p42_sharpe - holdout_sharpe) / max(abs(p42_sharpe), 1e-12)
    if drop <= DEGRADATION_STABLE_MAX and holdout_sharpe > 0:
        return "STABLE"
    if drop <= DEGRADATION_MODERATE_MAX and holdout_sharpe > 0:
        return "MODERATE_DEGRADATION"
    if holdout_sharpe > 0:
        return "SEVERE_DEGRADATION"
    return "FAILED_REPLICATION"


__all__ = [
    "DISCLAIMER",
    "FinalHoldoutConfig",
    "FROZEN_CANDIDATE_IDS",
    "PAPER_GATE_REQUIRED",
    "classify_degradation",
]
