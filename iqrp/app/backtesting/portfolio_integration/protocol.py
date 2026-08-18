"""Prompt 41 — Portfolio construction integration protocol (frozen).

Thin integration only. Does not rebuild PortfolioOptimizer implementations.
Prompt 35/36/39/40 artifacts are immutable inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

DISCLAIMER = (
    "PORTFOLIO CONSTRUCTION INTEGRATION — wiring/validation only. "
    "PORTFOLIO IMPLEMENTED ≠ INTEGRATED ≠ ROBUST ≠ PROFITABLE ≠ LIVE READY. "
    "Prompt 35/36/39/40 artifacts are immutable."
)

INTEGRATION_ID = "portfolio_construction_integration_v1"
SOFTWARE_VERSION = "iqrp-portfolio-construction-integration-0.1.0"
RANDOM_SEED = 41
PROMPT40_DIR = "results/candidate_consolidation"
PROMPT39_DIR = "results/model_driven_alpha_campaign"
OUTPUT_DIR = "results/portfolio_construction_integration"

# Methods to exercise (existing APIs only)
METHODS = (
    "mean_variance",
    "risk_parity",
    "black_litterman",
    "hrp",
    "constraints_only",
)

# Causal estimation: train+validation only for mu/cov; OOS for evaluation
TRAIN_FRAC = 0.50
VALIDATION_FRAC = 0.25

# Portfolio constraint defaults (research sim; not live)
MAX_WEIGHT = 0.35
MAX_GROSS = 1.0
MAX_NET = 1.0
MAX_TURNOVER = 0.75
RISK_AVERSION = 1.0
BUDGET = 1.0

COST_NAMES = ("BASE", "MODERATE", "ADVERSE")


@dataclass
class PortfolioIntegrationConfig:
    integration_id: str = INTEGRATION_ID
    prompt40_dir: str = PROMPT40_DIR
    prompt39_dir: str = PROMPT39_DIR
    output_dir: str = OUTPUT_DIR
    registry_path: str = "dataset_registry.json"
    random_seed: int = RANDOM_SEED
    software_version: str = SOFTWARE_VERSION
    methods: tuple[str, ...] = METHODS
    train_frac: float = TRAIN_FRAC
    validation_frac: float = VALIDATION_FRAC
    max_weight: float = MAX_WEIGHT
    max_gross: float = MAX_GROSS
    max_net: float = MAX_NET
    max_turnover: float = MAX_TURNOVER
    risk_aversion: float = RISK_AVERSION
    budget: float = BUDGET
    market_type: str = "crypto"
    smoke: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["disclaimer"] = DISCLAIMER
        d["causal_policy"] = (
            "mu and cov estimated from train+validation daily candidate net returns only. "
            "OOS returns used solely for evaluation. Future realized returns never enter "
            "weight construction at the decision timestamp."
        )
        d["signed_exposure_policy"] = (
            "Optimizer sleeve weights (often non-negative for RP/HRP) are multiplied by "
            "each candidate's directional signal sign to form signed trading exposure. "
            "RP/HRP remain long-only on sleeve budgets; they are NOT converted into "
            "directional long-only trading by silently dropping shorts."
        )
        d["methods"] = list(self.methods)
        d["cost_names"] = list(COST_NAMES)
        return d


__all__ = [
    "COST_NAMES",
    "DISCLAIMER",
    "INTEGRATION_ID",
    "METHODS",
    "PortfolioIntegrationConfig",
]
