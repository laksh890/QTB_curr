"""Prompt 40 — Candidate consolidation protocol (frozen before analysis).

Thresholds and formulas declared a priori. Do not alter after seeing results.
Research evidence is not a profitability guarantee. Prompt 39 artifacts are immutable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

DISCLAIMER = (
    "CANDIDATE CONSOLIDATION RESEARCH — predefined protocol. "
    "Research evidence is not a profitability guarantee. "
    "DISTINCT_RESEARCH_CANDIDATE ≠ PROFITABLE ≠ PRODUCTION_READY ≠ LIVE_READY. "
    "Prompt 39 artifacts are immutable inputs."
)

CONSOLIDATION_ID = "candidate_consolidation_v1"
SOFTWARE_VERSION = "iqrp-candidate-consolidation-0.1.0"
RANDOM_SEED = 40
PROMPT39_DIR = "results/model_driven_alpha_campaign"
OUTPUT_DIR = "results/candidate_consolidation"

# --- Dependence / redundancy (declared before selection) ---
# Primary classification uses VALIDATION-period |Pearson| of daily net returns.
CORR_HIGHLY_REDUNDANT = 0.85
CORR_RELATED = 0.50
# Cluster merge: connected components if |val Pearson daily net| >= this
CLUSTER_MERGE_CORR = 0.70

# --- Trade frequency buckets (trades/day) ---
FREQ_LOW = 1.0
FREQ_MODERATE = 5.0
FREQ_HIGH = 20.0
# Exclusion from DISTINCT_RESEARCH_CANDIDATES
OVERTRADING_TRADES_PER_DAY = 10.0

# --- OOS stability (compare val vs OOS net Sharpe; not a discard rule alone) ---
STABILITY_DEGRADED_RATIO = 0.50  # OOS sharpe < 50% of val sharpe (when val>0)

# --- Confidence weights (VALIDATION ONLY; formula fixed) ---
# w_i = max(validation_net_sharpe_i, 0) + CONF_EPS; then normalize.
CONF_EPS = 0.05

COST_NAMES = ("BASE", "MODERATE", "ADVERSE")
ENSEMBLE_METHODS = ("equal_weight", "confidence_weighted", "regime_conditioned", "majority_vote")


@dataclass
class ConsolidationConfig:
    consolidation_id: str = CONSOLIDATION_ID
    prompt39_dir: str = PROMPT39_DIR
    output_dir: str = OUTPUT_DIR
    registry_path: str = "dataset_registry.json"
    random_seed: int = RANDOM_SEED
    software_version: str = SOFTWARE_VERSION
    corr_highly_redundant: float = CORR_HIGHLY_REDUNDANT
    corr_related: float = CORR_RELATED
    cluster_merge_corr: float = CLUSTER_MERGE_CORR
    freq_low: float = FREQ_LOW
    freq_moderate: float = FREQ_MODERATE
    freq_high: float = FREQ_HIGH
    overtrading_trades_per_day: float = OVERTRADING_TRADES_PER_DAY
    stability_degraded_ratio: float = STABILITY_DEGRADED_RATIO
    conf_eps: float = CONF_EPS
    market_type: str = "crypto"
    timezone: str = "UTC"
    smoke: bool = False
    max_candidates_smoke: int = 12

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["disclaimer"] = DISCLAIMER
        d["ensemble_methods"] = list(ENSEMBLE_METHODS)
        d["cost_names"] = list(COST_NAMES)
        d["weighting_formula"] = (
            "confidence: w_i = max(validation_net_sharpe_i, 0) + conf_eps; normalize; "
            "NO OOS metrics enter weights."
        )
        d["clustering_method"] = (
            "Connected components on validation |Pearson| of calendar-daily net returns "
            f">= cluster_merge_corr ({self.cluster_merge_corr}). "
            "Each candidate's daily series is split chronologically 50/25/25; pairwise "
            "validation correlation uses the intersection of those validation dates. "
            "Not optimized against OOS profitability."
        )
        d["redundancy_rules"] = {
            "HIGHLY_REDUNDANT": f"|val_pearson_daily_net| >= {self.corr_highly_redundant}",
            "RELATED": f"{self.corr_related} <= |val_pearson_daily_net| < {self.corr_highly_redundant}",
            "DISTINCT": f"|val_pearson_daily_net| < {self.corr_related}",
        }
        d["representative_rule"] = (
            "One representative per cluster: highest VALIDATION net Sharpe "
            "(ties: lower trades/day, then experiment_id). OOS not used for selection."
        )
        d["distinct_research_candidate_rule"] = (
            "Cluster representatives that: reconstruct OK; trades/day <= overtrading threshold; "
            "not pairwise HIGHLY_REDUNDANT to an already-kept rep (val corr); "
            "BASE cost survival already implied by P39 CANDIDATE. "
            "OOS metrics recorded for evaluation only — not used to maximize Sharpe."
        )
        return d


__all__ = [
    "CONSOLIDATION_ID",
    "COST_NAMES",
    "CORR_HIGHLY_REDUNDANT",
    "CORR_RELATED",
    "CLUSTER_MERGE_CORR",
    "DISCLAIMER",
    "ENSEMBLE_METHODS",
    "ConsolidationConfig",
]
