"""Independent OOS validation of frozen Prompt-42 candidates.

Strict temporal firewall vs Prompts 35–42. No retuning. No LIVE_READY.
Calendar-duration gates dominate bar-count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DISCLAIMER = (
    "INDEPENDENT CANDIDATE VALIDATION — frozen alphas only. "
    "INSUFFICIENT_HOLDOUT / INVALID_HOLDOUT ≠ profitability. "
    "ROBUST_RESEARCH_EVIDENCE ≠ PAPER READY ≠ LIVE READY. "
    "Do not retune. Do not purchase data. Do not connect a broker."
)

VALIDATION_ID = "independent_candidate_validation_v1"
SOFTWARE_VERSION = "iqrp-independent-candidate-validation-0.1.0"
OUTPUT_DIR = "results/independent_candidate_validation"

# Frozen eligible + negative control
FROZEN_PRIMARY: tuple[str, ...] = (
    "mdc_99aa952c5d5f6ff7",
    "mdc_678609c534d68189",
)
NEGATIVE_CONTROL_ID = "mdc_6f008c954ea26bf5"
FROZEN_ALL: tuple[str, ...] = FROZEN_PRIMARY + (NEGATIVE_CONTROL_ID,)

# Calendar-duration gates (NOT bar-count)
MIN_DAYS_PROFITABILITY_INFERENCE = 30
MIN_DAYS_SUFFICIENT = 180
PREFERRED_DAYS = 365

# Paper-trading gate (all required)
PAPER_GATE_REQUIRED: tuple[str, ...] = (
    "holdout_calendar_days_ge_180",
    "positive_net_oos",
    "multiple_independent_periods",
    "acceptable_drawdown",
    "survives_BASE",
    "survives_MODERATE",
    "not_regime_concentrated",
    "statistically_credible",
    "reproducible",
    "no_leakage",
    "no_temporal_overlap",
    "candidate_immutable",
)


@dataclass
class IndependentValidationConfig:
    validation_id: str = VALIDATION_ID
    output_dir: str = OUTPUT_DIR
    prompt39_dir: str = "results/model_driven_alpha_campaign"
    prompt42_dir: str = "results/final_trading_validation"
    registry_path: str = "dataset_registry.json"
    frozen_primary: tuple[str, ...] = FROZEN_PRIMARY
    negative_control_id: str = NEGATIVE_CONTROL_ID
    random_seed: int = 42
    software_version: str = SOFTWARE_VERSION
    market_type: str = "crypto"
    warmup_bars: dict[str, int] = field(
        default_factory=lambda: {"5m": 5000, "15m": 3000, "30m": 2000, "1h": 1500}
    )
    n_walk_forward: int = 4
    n_bootstrap: int = 200
    block_len_bars: int = 48

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["disclaimer"] = DISCLAIMER
        d["live_ready"] = False
        d["paper_gate_required"] = list(PAPER_GATE_REQUIRED)
        d["calendar_gates"] = {
            "invalid_below_days": MIN_DAYS_PROFITABILITY_INFERENCE,
            "insufficient_below_days": MIN_DAYS_SUFFICIENT,
            "preferred_days": PREFERRED_DAYS,
            "note": "Do not compensate short calendar duration with bar count.",
        }
        d["anti_post_hoc"] = (
            "Frozen definitions immutable after seeing validation results. "
            "Nearby holding/TF diagnostics are non-selective."
        )
        return d


def classify_holdout_duration(calendar_days: int) -> str:
    if calendar_days < MIN_DAYS_PROFITABILITY_INFERENCE:
        return "INVALID_HOLDOUT"
    if calendar_days < MIN_DAYS_SUFFICIENT:
        return "INSUFFICIENT_HOLDOUT"
    return "DURATION_ADEQUATE"


def classify_candidate(
    *,
    duration_status: str,
    net_sharpe: float | None,
    survives_base: bool,
    survives_moderate: bool,
    regime_ok: bool,
    stat_ok: bool,
    is_negative_control: bool = False,
) -> str:
    """Final per-candidate class. Duration inadequacy dominates."""
    if duration_status == "INVALID_HOLDOUT":
        return "INVALID_HOLDOUT"
    if duration_status == "INSUFFICIENT_HOLDOUT":
        return "INSUFFICIENT_HOLDOUT"
    # Adequate duration path
    if net_sharpe is None or not (net_sharpe == net_sharpe):
        return "FAILED_REPLICATION"
    if not survives_base or (net_sharpe is not None and net_sharpe <= 0):
        return "FAILED_REPLICATION"
    if is_negative_control and survives_base and net_sharpe > 0:
        return "WEAK_EVIDENCE"  # unexpected; document, do not promote
    if not survives_moderate or not regime_ok:
        return "WEAK_EVIDENCE"
    if not stat_ok:
        return "PROMISING_OOS"
    return "ROBUST_RESEARCH_EVIDENCE"


__all__ = [
    "DISCLAIMER",
    "IndependentValidationConfig",
    "FROZEN_PRIMARY",
    "NEGATIVE_CONTROL_ID",
    "FROZEN_ALL",
    "PAPER_GATE_REQUIRED",
    "classify_holdout_duration",
    "classify_candidate",
]
