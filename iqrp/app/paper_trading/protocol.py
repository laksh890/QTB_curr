"""Prompt 43 — Paper trading & realistic execution validation protocol.

Frozen candidates immutable. No 2025 retuning. No LIVE_READY / PROVEN PROFITABLE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DISCLAIMER = (
    "PAPER TRADING VALIDATION — sequential simulated execution only. "
    "PAPER_TRADING_CANDIDATE ≠ PROVEN PROFITABLE ≠ LIVE_READY ≠ PRODUCTION_READY. "
    "Costs/spreads are ASSUMED when bid/ask unavailable. No broker connection."
)

VALIDATION_ID = "paper_trading_validation_v1"
SOFTWARE_VERSION = "iqrp-paper-trading-validation-0.1.0"
OUTPUT_DIR = "results/paper_trading_validation"

FROZEN_CANDIDATES: dict[str, str] = {
    "A": "mdc_99aa952c5d5f6ff7",
    "B": "mdc_6f008c954ea26bf5",
    "C": "mdc_678609c534d68189",
}

RESEARCH_END = "2024-12-31 23:59:59+00:00"
HOLDOUT_START = "2025-01-01 00:00:00+00:00"
HOLDOUT_END = "2025-12-31 23:59:59+00:00"

# Assumed microstructure (OHLCV — not observed bid/ask)
EXEC_SCENARIOS: dict[str, dict[str, float]] = {
    "BASE": {
        "commission_bps": 1.0,
        "half_spread_bps": 1.0,  # half-spread paid on market orders
        "slippage_bps": 2.0,
        "latency_bars": 1.0,
        "partial_fill_prob": 0.02,
        "reject_prob": 0.005,
        "variable_spread_bps": 0.5,
    },
    "MODERATE": {
        "commission_bps": 2.0,
        "half_spread_bps": 2.0,
        "slippage_bps": 4.0,
        "latency_bars": 1.0,
        "partial_fill_prob": 0.05,
        "reject_prob": 0.01,
        "variable_spread_bps": 1.0,
    },
    "ADVERSE": {
        "commission_bps": 4.0,
        "half_spread_bps": 4.0,
        "slippage_bps": 8.0,
        "latency_bars": 2.0,
        "partial_fill_prob": 0.10,
        "reject_prob": 0.03,
        "variable_spread_bps": 2.0,
    },
}

COMBOS: tuple[tuple[str, ...], ...] = (
    ("A",),
    ("B",),
    ("C",),
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
    ("A", "B", "C"),
)


@dataclass
class PaperTradingValidationConfig:
    validation_id: str = VALIDATION_ID
    output_dir: str = OUTPUT_DIR
    prompt39_dir: str = "results/model_driven_alpha_campaign"
    frozen_2025_dir: str = "results/frozen_2024_2025_holdout"
    data_dir: str = "data/btcusdt/firewall_2024_2025"
    random_seed: int = 43
    software_version: str = SOFTWARE_VERSION
    initial_capital: float = 100_000.0
    max_position: float = 0.20
    max_gross: float = 1.0
    max_net: float = 1.0
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.25
    max_turnover_per_bar: float = 0.50
    sleeve_weight: float = 0.33  # per-candidate target abs weight in combos
    exec_scenario: str = "BASE"
    smoke: bool = False
    smoke_bars: int = 500
    run_failure_injection: bool = True
    market_type: str = "crypto"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["disclaimer"] = DISCLAIMER
        d["frozen_candidates"] = dict(FROZEN_CANDIDATES)
        d["research_end"] = RESEARCH_END
        d["holdout"] = [HOLDOUT_START, HOLDOUT_END]
        d["exec_scenarios"] = EXEC_SCENARIOS
        d["combos"] = [list(c) for c in COMBOS]
        d["live_ready"] = False
        d["bid_ask_label"] = "ASSUMED_OHLCV_MICROSTRUCTURE"
        return d


def classify_paper_status(gates: dict[str, bool]) -> str:
    required = [
        "candidates_frozen",
        "no_lookahead",
        "sequential_ok",
        "execution_model_ok",
        "recon_zero_drift",
        "fills_to_positions_ok",
        "fees_accounted",
        "risk_limits_enforced",
        "kill_switches_ok",
        "failure_injection_ok",
        "no_2025_retune",
        "reproducible",
    ]
    failed = [k for k in required if not gates.get(k)]
    if failed:
        if gates.get("sequential_ok") and gates.get("execution_model_ok"):
            return "PAPER_VALIDATION_WEAK"
        return "PAPER_SIMULATION_OPERATIONAL" if gates.get("sequential_ok") else "PAPER_VALIDATION_WEAK"
    # All operational gates pass
    if gates.get("paper_pnl_positive_base") and gates.get("survivors_exist"):
        return "PAPER_TRADING_CANDIDATE"
    return "PAPER_VALIDATION_PASS"


__all__ = [
    "DISCLAIMER",
    "FROZEN_CANDIDATES",
    "EXEC_SCENARIOS",
    "COMBOS",
    "PaperTradingValidationConfig",
    "classify_paper_status",
]
