"""Predeclared Prompt 39 model-driven research protocol.

Frozen before any OOS evaluation. Do not alter after seeing results.
Research evidence is not a profitability guarantee.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from iqrp.app.backtesting.alpha_research.types import COST_SCENARIOS, DEFAULT_FORWARD_HORIZONS

CAMPAIGN_ID = "model_driven_alpha_campaign_v1"
SOFTWARE_VERSION = "iqrp-model-driven-alpha-0.1.0"
DISCLAIMER = (
    "MODEL-DRIVEN ALPHA RESEARCH — predefined protocol. "
    "Research evidence is not a profitability guarantee. "
    "MODEL IMPLEMENTED ≠ FORECAST ≠ SIGNAL ≠ OOS ≠ COST-SURVIVAL ≠ ROBUST ≠ PROFITABLE ≠ LIVE-READY."
)
RANDOM_SEED = 39

# Research subsample of registered history ending 2024-12-31 (do not fabricate later data).
MAX_BARS: dict[str, int] = {
    "1m": 40_000,
    "5m": 30_000,
    "15m": 25_000,
    "30m": 20_000,
    "1h": 20_000,
}

DATASET_KEYS: dict[str, str] = {
    "1m": "btcusdt_intraday_1m@1.0.0",
    "5m": "btcusdt_intraday_5m@1.0.0",
    "15m": "btcusdt_intraday_15m@1.0.0",
    "30m": "btcusdt_intraday_30m@1.0.0",
    "1h": "btcusdt_intraday_1h@1.0.0",
}

TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h")
HOLDING_BARS: tuple[int, ...] = DEFAULT_FORWARD_HORIZONS
DIRECTIONS: tuple[str, ...] = ("LONG", "SHORT", "LONG_SHORT")
COST_NAMES: tuple[str, ...] = ("BASE", "MODERATE", "ADVERSE")
REFERENCE_LOOKBACK = 20
TRAIN_FRAC = 0.50
VALIDATION_FRAC = 0.25

REFERENCE_SIGNALS: tuple[str, ...] = (
    "momentum_signal",
    "mean_reversion_signal",
    "breakout_signal",
    "trend_signal",
    "volatility_signal",
    "volume_signal",
    "price_action_signal",
)

# Declared model availability by timeframe. UNAVAILABLE elsewhere with reason.
MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "family": "GARCH",
        "model_id": "garch",
        "adapter_id": "garch_volatility_v1",
        "pipeline": "volatility",
        "timeframes": ("5m", "15m", "30m", "1h"),
        "unavailable": {"1m": "GARCH MLE on 1m research window exceeds campaign compute budget"},
    },
    {
        "family": "ARIMA",
        "model_id": "arima",
        "adapter_id": "arima_return_v1",
        "pipeline": "statistical",
        "timeframes": ("15m", "30m", "1h"),
        "unavailable": {
            "1m": "ARIMA unfit for dense 1m campaign subsample",
            "5m": "ARIMA reserved for coarser TFs in declared protocol",
        },
    },
    {
        "family": "VAR",
        "model_id": "var",
        "adapter_id": None,
        "pipeline": None,
        "timeframes": (),
        "unavailable": {tf: "Single-asset BTCUSDT — VAR requires multivariate series" for tf in TIMEFRAMES},
    },
    {
        "family": "VECM",
        "model_id": "vecm",
        "adapter_id": None,
        "pipeline": None,
        "timeframes": (),
        "unavailable": {tf: "Single-asset BTCUSDT — VECM requires cointegrated multivariate series" for tf in TIMEFRAMES},
    },
    {
        "family": "XGBoost",
        "model_id": "xgboost",
        "adapter_id": "xgb_return_v1",
        "pipeline": "tree_ml",
        "timeframes": ("15m", "30m", "1h"),
        "unavailable": {
            "1m": "Tree ML on 1m excluded by protocol compute budget",
            "5m": "Tree ML reserved for >=15m in declared protocol",
        },
    },
    {
        "family": "LightGBM",
        "model_id": "lightgbm",
        "adapter_id": "lgbm_return_v1",
        "pipeline": "tree_ml",
        "timeframes": ("15m", "30m", "1h"),
        "unavailable": {
            "1m": "Tree ML on 1m excluded by protocol compute budget",
            "5m": "Tree ML reserved for >=15m in declared protocol",
        },
    },
    {
        "family": "CatBoost",
        "model_id": "catboost",
        "adapter_id": "cat_return_v1",
        "pipeline": "tree_ml",
        "timeframes": ("15m", "30m", "1h"),
        "unavailable": {
            "1m": "Tree ML on 1m excluded by protocol compute budget",
            "5m": "Tree ML reserved for >=15m in declared protocol",
        },
    },
    {
        "family": "LSTM",
        "model_id": "lstm",
        "adapter_id": "lstm_return_v1",
        "pipeline": "neural",
        "timeframes": ("30m", "1h"),
        "unavailable": {
            "1m": "Neural training excluded on fine TFs by protocol",
            "5m": "Neural training excluded on fine TFs by protocol",
            "15m": "Neural training reserved for >=30m in declared protocol",
        },
    },
    {
        "family": "GRU",
        "model_id": "gru",
        "adapter_id": "gru_return_v1",
        "pipeline": "neural",
        "timeframes": ("1h",),
        "unavailable": {tf: "GRU reserved for 1h in declared protocol" for tf in ("1m", "5m", "15m", "30m")},
    },
    {
        "family": "MLP",
        "model_id": "mlp",
        "adapter_id": "mlp_return_v1",
        "pipeline": "neural",
        "timeframes": ("1h",),
        "unavailable": {tf: "MLP reserved for 1h in declared protocol" for tf in ("1m", "5m", "15m", "30m")},
    },
    {
        "family": "Transformer",
        "model_id": "tide",
        "adapter_id": "transformer_return_v1",
        "pipeline": "transformer",
        "timeframes": ("1h",),
        "unavailable": {tf: "Transformer reserved for 1h in declared protocol" for tf in ("1m", "5m", "15m", "30m")},
    },
    {
        "family": "HMM",
        "model_id": "hmm",
        "adapter_id": "hmm_regime_v1",
        "pipeline": "regime",
        "timeframes": ("15m", "30m", "1h"),
        "unavailable": {
            "1m": "HMM excluded on 1m by protocol compute budget",
            "5m": "HMM reserved for >=15m in declared protocol",
        },
    },
    {
        "family": "GMM",
        "model_id": "gmm",
        "adapter_id": None,
        "pipeline": None,
        "timeframes": (),
        "unavailable": {
            tf: "GMM regime model present in package but not registered in default adapter/pipeline path"
            for tf in TIMEFRAMES
        },
    },
    {
        "family": "Markov",
        "model_id": "markov_chain",
        "adapter_id": "markov_regime_v1",
        "pipeline": "regime",
        "timeframes": ("1h",),
        "unavailable": {tf: "Markov reserved for 1h smoke in declared protocol" for tf in ("1m", "5m", "15m", "30m")},
    },
)

# Economically meaningful combinations (AND agreement). Declared a priori.
COMBINATIONS: tuple[dict[str, Any], ...] = (
    {"id": "garch_x_momentum", "model_adapter": "garch_volatility_v1", "reference": "momentum_signal", "timeframes": ("1h", "30m")},
    {"id": "garch_x_breakout", "model_adapter": "garch_volatility_v1", "reference": "breakout_signal", "timeframes": ("1h",)},
    {"id": "garch_x_meanrev", "model_adapter": "garch_volatility_v1", "reference": "mean_reversion_signal", "timeframes": ("1h",)},
    {"id": "hmm_x_momentum", "model_adapter": "hmm_regime_v1", "reference": "momentum_signal", "timeframes": ("1h", "30m")},
    {"id": "hmm_x_meanrev", "model_adapter": "hmm_regime_v1", "reference": "mean_reversion_signal", "timeframes": ("1h",)},
    {"id": "hmm_x_breakout", "model_adapter": "hmm_regime_v1", "reference": "breakout_signal", "timeframes": ("1h",)},
    {"id": "arima_x_price_action", "model_adapter": "arima_return_v1", "reference": "price_action_signal", "timeframes": ("1h",)},
    {"id": "arima_x_momentum", "model_adapter": "arima_return_v1", "reference": "momentum_signal", "timeframes": ("1h",)},
    {"id": "xgb_x_momentum", "model_adapter": "xgb_return_v1", "reference": "momentum_signal", "timeframes": ("1h", "30m")},
    {"id": "xgb_x_volatility", "model_adapter": "xgb_return_v1", "reference": "volatility_signal", "timeframes": ("1h",)},
    {"id": "xgb_x_price_action", "model_adapter": "xgb_return_v1", "reference": "price_action_signal", "timeframes": ("1h",)},
    {"id": "transformer_x_momentum", "model_adapter": "transformer_return_v1", "reference": "momentum_signal", "timeframes": ("1h",)},
    {"id": "transformer_x_volatility", "model_adapter": "transformer_return_v1", "reference": "volatility_signal", "timeframes": ("1h",)},
)

# Causal MTF pairs: model_tf → execution_tf
MTF_PAIRS: tuple[dict[str, Any], ...] = (
    {"model_tf": "1h", "exec_tf": "15m", "sources": ("garch_volatility_v1", "arima_return_v1", "xgb_return_v1", "hmm_regime_v1", "momentum_signal")},
    {"model_tf": "30m", "exec_tf": "5m", "sources": ("garch_volatility_v1", "momentum_signal")},
    {"model_tf": "15m", "exec_tf": "5m", "sources": ("momentum_signal", "xgb_return_v1")},
)

# Ensembles: weights fixed a priori (no OOS fitting)
ENSEMBLES: tuple[dict[str, Any], ...] = (
    {
        "id": "equal_weight_ref_1h",
        "method": "equal_weight",
        "timeframe": "1h",
        "members": ("momentum_signal", "trend_signal", "breakout_signal"),
        "weights": (1 / 3, 1 / 3, 1 / 3),
    },
    {
        "id": "majority_vote_ref_1h",
        "method": "majority_vote",
        "timeframe": "1h",
        "members": ("momentum_signal", "trend_signal", "breakout_signal"),
        "weights": None,
    },
    {
        "id": "confidence_weight_model_1h",
        "method": "confidence_weighted",
        "timeframe": "1h",
        "members": ("arima_return_v1", "xgb_return_v1", "momentum_signal"),
        # Fixed a priori confidence weights (not OOS-tuned)
        "weights": (0.25, 0.35, 0.40),
    },
    {
        "id": "regime_conditioned_mom_1h",
        "method": "regime_conditioned",
        "timeframe": "1h",
        "members": ("hmm_regime_v1", "momentum_signal"),
        "weights": None,
    },
)


@dataclass
class ModelCampaignConfig:
    campaign_id: str = CAMPAIGN_ID
    output_dir: str = "results/model_driven_alpha_campaign"
    registry_path: str = "dataset_registry.json"
    dataset_keys: dict[str, str] = field(default_factory=lambda: dict(DATASET_KEYS))
    timeframes: tuple[str, ...] = TIMEFRAMES
    holding_bars: tuple[int, ...] = HOLDING_BARS
    directions: tuple[str, ...] = DIRECTIONS
    cost_scenarios: tuple[str, ...] = COST_NAMES
    max_bars: dict[str, int] = field(default_factory=lambda: dict(MAX_BARS))
    train_frac: float = TRAIN_FRAC
    validation_frac: float = VALIDATION_FRAC
    random_seed: int = RANDOM_SEED
    reference_lookback: int = REFERENCE_LOOKBACK
    market_type: str = "crypto"
    timezone: str = "UTC"
    software_version: str = SOFTWARE_VERSION
    # Test/smoke override: limit timeframes / holdings
    smoke: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["disclaimer"] = DISCLAIMER
        d["cost_scenario_defs"] = {k: dict(COST_SCENARIOS[k]) for k in self.cost_scenarios}
        d["model_specs"] = list(MODEL_SPECS)
        d["combinations"] = list(COMBINATIONS)
        d["mtf_pairs"] = list(MTF_PAIRS)
        d["ensembles"] = list(ENSEMBLES)
        d["reference_signals"] = list(REFERENCE_SIGNALS)
        return d


def apply_direction_mask(signal, direction: str):
    import numpy as np
    import pandas as pd

    s = pd.Series(signal).astype(float).fillna(0.0)
    if direction == "LONG":
        return s.clip(lower=0.0)
    if direction == "SHORT":
        return s.clip(upper=0.0)
    if direction == "LONG_SHORT":
        return s
    raise ValueError(direction)


def combine_and_agree(a, b):
    import numpy as np
    import pandas as pd

    aa = pd.Series(a).fillna(0.0).to_numpy(dtype=float)
    bb = pd.Series(b).fillna(0.0).to_numpy(dtype=float)
    out = np.zeros(len(aa), dtype=float)
    out[(aa > 0) & (bb > 0)] = 1.0
    out[(aa < 0) & (bb < 0)] = -1.0
    return pd.Series(out, index=pd.Series(a).index)


__all__ = [
    "CAMPAIGN_ID",
    "COMBINATIONS",
    "COST_NAMES",
    "DATASET_KEYS",
    "DIRECTIONS",
    "DISCLAIMER",
    "ENSEMBLES",
    "HOLDING_BARS",
    "MAX_BARS",
    "MODEL_SPECS",
    "MTF_PAIRS",
    "ModelCampaignConfig",
    "REFERENCE_SIGNALS",
    "SOFTWARE_VERSION",
    "TIMEFRAMES",
    "apply_direction_mask",
    "combine_and_agree",
]
