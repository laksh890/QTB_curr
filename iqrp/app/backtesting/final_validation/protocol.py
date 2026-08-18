"""Prompt 42 — Final trading validation protocol (frozen before evaluation).

Do not alter gates after seeing OOS results. Do not manufacture profitability.
Maximum claim: PROFITABILITY_EVIDENCE (not LIVE_READY / not absolute proven profit).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DISCLAIMER = (
    "FINAL TRADING VALIDATION — research evidence only. "
    "PROFITABILITY_EVIDENCE ≠ PROVEN PROFITABILITY ≠ PAPER-READY ≠ LIVE_READY. "
    "Do not p-hack: gates and grids are predefined. Prompt 35/36/39/40 artifacts immutable."
)

VALIDATION_ID = "final_trading_validation_v1"
SOFTWARE_VERSION = "iqrp-final-trading-validation-0.1.0"
RANDOM_SEED = 42
OUTPUT_DIR = "results/final_trading_validation"
PROMPT40_DIR = "results/candidate_consolidation"
PROMPT39_DIR = "results/model_driven_alpha_campaign"
PROMPT41_DIR = "results/portfolio_construction_integration"

# Prefer extended registry versions when present; fall back to @1.0.0
DATASET_KEYS_V101: dict[str, str] = {
    "1m": "btcusdt_intraday_1m@1.0.1",
    "5m": "btcusdt_intraday_5m@1.0.1",
    "15m": "btcusdt_intraday_15m@1.0.1",
    "30m": "btcusdt_intraday_30m@1.0.1",
    "1h": "btcusdt_intraday_1h@1.0.1",
}
DATASET_KEYS_V100: dict[str, str] = {
    "1m": "btcusdt_intraday_1m@1.0.0",
    "5m": "btcusdt_intraday_5m@1.0.0",
    "15m": "btcusdt_intraday_15m@1.0.0",
    "30m": "btcusdt_intraday_30m@1.0.0",
    "1h": "btcusdt_intraday_1h@1.0.0",
}

# Research subsample of available history (chronological; OOS = last segment)
MAX_BARS: dict[str, int] = {
    "1m": 120_000,
    "5m": 80_000,
    "15m": 50_000,
    "30m": 40_000,
    "1h": 40_000,
    "4h": 20_000,
}

TIMEFRAMES: tuple[str, ...] = ("5m", "15m", "30m", "1h")
HOLDING_BARS: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 30, 60)
DIRECTIONS: tuple[str, ...] = ("LONG", "SHORT", "LONG_SHORT")
COST_NAMES: tuple[str, ...] = ("BASE", "MODERATE", "ADVERSE")

TRAIN_FRAC = 0.50
VALIDATION_FRAC = 0.25

# --- Profitability gate (ALL required for PROFITABILITY_EVIDENCE) ---
GATE_MIN_OOS_NET_RETURN = 0.0
GATE_MIN_OOS_SHARPE = 0.0
GATE_MIN_EXPECTANCY = 0.0
GATE_MIN_TRADES = 30
GATE_MAX_DD = 0.50  # absolute fraction
GATE_MIN_PERTURB_SURVIVAL = 0.5  # fraction of neighbor params that stay net-positive OOS
GATE_ADVERSE_CATASTROPHIC_SHARPE = -2.0  # below this = catastrophic ADVERSE failure

# Trade-behavior classification thresholds (trades/day)
BEHAVIOR_BUY_HOLD_MAX = 0.05
BEHAVIOR_LOW_FREQ_MAX = 0.5
BEHAVIOR_SWING_MAX = 2.0
BEHAVIOR_INTRADAY_MAX = 10.0
BEHAVIOR_OVERTRADE_MIN = 50.0

# Free data source survey (predeclared classifications — evidence filled at runtime)
FREE_SOURCE_SURVEY: tuple[dict[str, Any], ...] = (
    {
        "provider": "Binance Vision",
        "url": "https://data.binance.vision/",
        "asset_class": "crypto_spot",
        "resolution": "1m+",
        "bid_ask": False,
        "depth": False,
        "license": "UNKNOWN (public CDN; verify ToS independently)",
        "grade_preliminary": "RESEARCH_GRADE",
        "notes": "Official exchange historical monthly klines; best free crypto OHLCV in-repo.",
    },
    {
        "provider": "Binance REST /api/v3/klines",
        "url": "https://api.binance.com",
        "asset_class": "crypto_spot",
        "resolution": "1m+",
        "bid_ask": False,
        "depth": False,
        "license": "UNKNOWN",
        "grade_preliminary": "RESEARCH_GRADE",
        "notes": "Useful for incomplete current month; not a full historical archive replacement.",
    },
    {
        "provider": "Yahoo Finance / yfinance",
        "url": "https://finance.yahoo.com/",
        "asset_class": "equities/indices/crypto_spot_proxy",
        "resolution": "1m limited (~7d), daily long",
        "bid_ask": False,
        "depth": False,
        "license": "UNKNOWN",
        "grade_preliminary": "DEVELOPMENT_GRADE",
        "notes": "Already in-repo for NIFTY; not suitable as primary BTC intraday research source.",
    },
    {
        "provider": "Stooq",
        "url": "https://stooq.com/",
        "asset_class": "equities/fx/crypto_limited",
        "resolution": "mostly daily",
        "bid_ask": False,
        "depth": False,
        "license": "UNKNOWN",
        "grade_preliminary": "DEVELOPMENT_GRADE",
        "notes": "Investigated; no superior free BTC 1m archive vs Binance Vision.",
    },
    {
        "provider": "Alpha Vantage",
        "url": "https://www.alphavantage.co/",
        "asset_class": "equities/fx/crypto",
        "resolution": "intraday with rate limits",
        "bid_ask": False,
        "depth": False,
        "license": "API ToS; free tier limited",
        "grade_preliminary": "DEVELOPMENT_GRADE",
        "notes": "Rate limits / history depth inferior for multi-year BTC 1m research.",
    },
    {
        "provider": "Nasdaq Data Link",
        "url": "https://data.nasdaq.com/",
        "asset_class": "mixed",
        "resolution": "varies",
        "bid_ask": "varies",
        "depth": False,
        "license": "mixed free/paid",
        "grade_preliminary": "UNKNOWN",
        "notes": "Free datasets do not clearly dominate Binance Vision for BTC spot 1m OHLCV.",
    },
    {
        "provider": "CME public / sample datasets",
        "url": "https://www.cmegroup.com/",
        "asset_class": "futures",
        "resolution": "varies",
        "bid_ask": "sometimes",
        "depth": False,
        "license": "restrictive; often paid for full history",
        "grade_preliminary": "PROFESSIONAL_GRADE",
        "notes": "Potentially higher grade for futures; full tick/depth typically paid. Not auto-purchased.",
    },
    {
        "provider": "FRED",
        "url": "https://fred.stlouisfed.org/",
        "asset_class": "macro",
        "resolution": "daily/monthly",
        "bid_ask": False,
        "depth": False,
        "license": "public domain / FRED terms",
        "grade_preliminary": "INSTITUTIONAL_GRADE",
        "notes": "Macro only — not a substitute for BTC OHLCV trading bars.",
    },
)

PAID_UPGRADE_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "provider": "Tardis.dev / Kaiko / CryptoTick",
        "what_adds": "Tick trades, L2 order book, true bid/ask, point-in-time depth",
        "approximate_cost": "Typically subscription; report before purchase — NOT purchased in Prompt 42",
        "limitation_solved": "OHLCV-only cost model; no authentic spread path; no market-depth realism",
        "action": "STOP_BEFORE_PURCHASE",
    },
)


@dataclass
class FinalValidationConfig:
    validation_id: str = VALIDATION_ID
    output_dir: str = OUTPUT_DIR
    prompt40_dir: str = PROMPT40_DIR
    prompt39_dir: str = PROMPT39_DIR
    prompt41_dir: str = PROMPT41_DIR
    registry_path: str = "dataset_registry.json"
    random_seed: int = RANDOM_SEED
    software_version: str = SOFTWARE_VERSION
    timeframes: tuple[str, ...] = TIMEFRAMES
    holding_bars: tuple[int, ...] = HOLDING_BARS
    directions: tuple[str, ...] = DIRECTIONS
    cost_scenarios: tuple[str, ...] = COST_NAMES
    max_bars: dict[str, int] = field(default_factory=lambda: dict(MAX_BARS))
    train_frac: float = TRAIN_FRAC
    validation_frac: float = VALIDATION_FRAC
    market_type: str = "crypto"
    smoke: bool = False
    # Primary: validate P40 distinct set deeply (no post-hoc selection)
    validate_p40_candidates: bool = True
    # Secondary grid disabled by default: Prompt 39 already registered the exhaustive grid;
    # this stage deep-validates the frozen Prompt 40 distinct set (anti-p-hacking).
    run_predeclared_grid: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["disclaimer"] = DISCLAIMER
        d["profitability_gate"] = {
            "min_oos_net_return": GATE_MIN_OOS_NET_RETURN,
            "min_oos_sharpe": GATE_MIN_OOS_SHARPE,
            "min_expectancy": GATE_MIN_EXPECTANCY,
            "min_trades": GATE_MIN_TRADES,
            "max_drawdown": GATE_MAX_DD,
            "min_perturb_survival": GATE_MIN_PERTURB_SURVIVAL,
            "adverse_catastrophic_sharpe": GATE_ADVERSE_CATASTROPHIC_SHARPE,
            "required_all": [
                "positive_oos_net_return",
                "positive_oos_sharpe",
                "positive_expectancy_after_costs",
                "survives_BASE",
                "survives_MODERATE",
                "not_catastrophic_ADVERSE",
                "leakage_ok",
                "recon_ok",
                "execution_timing_ok",
                "walk_forward_ok",
                "not_tiny_window",
                "acceptable_drawdown",
                "acceptable_turnover",
                "sufficient_trades",
                "no_oos_contamination_in_selection",
                "parameter_perturbation_ok",
                "regime_ok",
                "reproducible",
            ],
        }
        d["anti_p_hacking"] = (
            "P40 candidates were selected without OOS Sharpe maximization. "
            "This validation reuses that set and a predeclared grid. "
            "Gates frozen before evaluation. No cost/gate relaxation after results."
        )
        d["free_source_survey"] = list(FREE_SOURCE_SURVEY)
        d["paid_upgrade_candidates"] = list(PAID_UPGRADE_CANDIDATES)
        return d


def classify_behavior(trades_per_day: float) -> str:
    t = float(trades_per_day or 0)
    if t <= BEHAVIOR_BUY_HOLD_MAX:
        return "BUY_AND_HOLD"
    if t <= BEHAVIOR_LOW_FREQ_MAX:
        return "LOW_FREQUENCY"
    if t <= BEHAVIOR_SWING_MAX:
        return "SWING"
    if t <= BEHAVIOR_INTRADAY_MAX:
        return "INTRADAY"
    if t >= BEHAVIOR_OVERTRADE_MIN:
        return "OVERTRADING"
    return "HIGH_FREQUENCY_RESEARCH"


__all__ = [
    "DISCLAIMER",
    "DATASET_KEYS_V100",
    "DATASET_KEYS_V101",
    "FinalValidationConfig",
    "FREE_SOURCE_SURVEY",
    "classify_behavior",
]
