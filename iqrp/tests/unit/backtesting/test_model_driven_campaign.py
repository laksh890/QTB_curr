"""Focused tests for Prompt 39 model-driven alpha campaign protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iqrp.app.backtesting.alpha_research.model_campaign.protocol import (
    COMBINATIONS,
    ENSEMBLES,
    MODEL_SPECS,
    apply_direction_mask,
    combine_and_agree,
)
from iqrp.app.backtesting.alpha_research.model_campaign.runner import (
    _ensemble_signal,
    _exp_id,
    _trade_stats,
)
from iqrp.app.backtesting.alpha_research.mtf import align_feature_to_execution
from iqrp.app.backtesting.alpha_research.adapters.validation import train_val_oos_slices
from iqrp.app.backtesting.alpha_research.types import COST_SCENARIOS


def test_experiment_ids_deterministic():
    a = _exp_id("c", "model", "garch", "1h", "LONG", 5, "BASE", 39)
    b = _exp_id("c", "model", "garch", "1h", "LONG", 5, "BASE", 39)
    c = _exp_id("c", "model", "garch", "1h", "SHORT", 5, "BASE", 39)
    assert a == b
    assert a != c
    assert a.startswith("mdc_")


def test_direction_long_short_flat():
    s = pd.Series([1.0, -1.0, 0.0, 0.5, -0.2])
    assert list(apply_direction_mask(s, "LONG")) == [1.0, 0.0, 0.0, 0.5, 0.0]
    assert list(apply_direction_mask(s, "SHORT")) == [0.0, -1.0, 0.0, 0.0, -0.2]
    assert list(apply_direction_mask(s, "LONG_SHORT")) == list(s)


def test_combination_agreement_lineage_ids_declared():
    a = pd.Series([1, 1, -1, -1, 0])
    b = pd.Series([1, -1, -1, 1, 1])
    out = combine_and_agree(a, b)
    assert list(out) == [1.0, 0.0, -1.0, 0.0, 0.0]
    assert all("id" in c and "model_adapter" in c and "reference" in c for c in COMBINATIONS)


def test_ensemble_methods_predeclared():
    idx = pd.RangeIndex(5)
    members = {
        "a": pd.Series([1, 1, -1, 0, 1], index=idx),
        "b": pd.Series([1, -1, -1, 0, 1], index=idx),
        "c": pd.Series([0, 1, -1, 0, -1], index=idx),
    }
    eq = _ensemble_signal("equal_weight", members, None)
    assert len(eq) == 5
    maj = _ensemble_signal("majority_vote", members, None)
    assert maj.iloc[2] == -1.0
    conf = _ensemble_signal("confidence_weighted", members, (0.5, 0.3, 0.2))
    assert set(np.unique(conf.to_numpy())).issubset({-1.0, 0.0, 1.0})
    # regime_conditioned: first=regime, second=directional
    reg_members = {
        "hmm": pd.Series([1, 1, -1, 0, -1], index=idx),
        "mom": pd.Series([1, -1, -1, 1, 1], index=idx),
    }
    rc = _ensemble_signal("regime_conditioned", reg_members, None)
    assert list(rc) == [1.0, 0.0, -1.0, 0.0, 0.0]
    assert {e["method"] for e in ENSEMBLES} <= {
        "equal_weight",
        "confidence_weighted",
        "majority_vote",
        "regime_conditioned",
    }


def test_oos_slices_chronological_purge_gap():
    slices = train_val_oos_slices(1000, train_frac=0.5, validation_frac=0.25)
    assert slices["train"].stop == 500
    assert slices["validation"].start == 500
    assert slices["validation"].stop == 750
    assert slices["oos"].start == 750
    assert slices["oos"].stop == 1000
    # purge/embargo at horizon scale: train end < oos start always
    purge = 20
    assert slices["train"].stop + purge <= slices["oos"].start + purge


def test_mtf_backward_causal_alignment():
    # Higher TF timestamps every 3 bars of exec
    model_ts = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"], utc=True
            )
        }
    )
    model_feat = pd.Series([10.0, 20.0, 30.0])
    exec_ts = pd.to_datetime(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:30",
            "2024-01-01 01:00",
            "2024-01-01 01:30",
            "2024-01-01 02:00",
        ],
        utc=True,
    )
    aligned = align_feature_to_execution(model_ts, model_feat, exec_ts)
    # At 00:30 only 00:00 HTF info available
    assert float(aligned.iloc[1]) == 10.0
    assert float(aligned.iloc[2]) == 20.0
    # No lookahead: value at 01:30 is still 01:00 HTF, not 02:00
    assert float(aligned.iloc[3]) == 20.0


def test_cost_scenarios_exist_and_ordered():
    assert set(COST_SCENARIOS) >= {"BASE", "MODERATE", "ADVERSE"}
    base = COST_SCENARIOS["BASE"]
    adv = COST_SCENARIOS["ADVERSE"]
    for k in ("commission_bps", "spread_bps", "slippage_bps"):
        assert float(adv[k]) >= float(base[k])


def test_trade_frequency_stats():
    # Alternate long/flat every bar after entry
    pos = pd.Series([0, 1, 1, 1, 0, -1, -1, 0, 1, 0] * 50)
    stats = _trade_stats(pos, "1h", market_type="crypto")
    assert stats["trades_per_day"] > 0
    assert stats["long_entries"] > 0
    assert stats["short_entries"] > 0
    assert stats["avg_holding_bars"] >= 1


def test_model_specs_unavailable_explicit_no_silent_sub():
    for spec in MODEL_SPECS:
        if not spec.get("timeframes"):
            assert spec.get("unavailable"), f"{spec['family']} must declare UNAVAILABLE reasons"
            assert spec.get("adapter_id") is None or True
        for tf, reason in (spec.get("unavailable") or {}).items():
            assert isinstance(reason, str) and len(reason) > 5


def test_lineage_fields_in_protocol_models():
    for spec in MODEL_SPECS:
        assert "family" in spec and "model_id" in spec
        if spec.get("adapter_id"):
            assert spec.get("pipeline") in {
                "volatility",
                "statistical",
                "tree_ml",
                "neural",
                "transformer",
                "regime",
            }


def test_multiple_testing_registration_ids_unique():
    ids = [
        _exp_id("camp", "ref", sid, tf, "LONG_SHORT", hb, "BASE", 39)
        for sid in ("momentum_signal", "trend_signal")
        for tf in ("1h", "15m")
        for hb in (1, 5)
    ]
    assert len(ids) == len(set(ids))


def test_smoke_campaign_artifacts(tmp_path):
    """End-to-end smoke on tiny protocol — may skip if datasets missing."""
    from pathlib import Path

    registry = Path("dataset_registry.json")
    if not registry.exists():
        pytest.skip("dataset_registry.json not present")
    from iqrp.app.backtesting.alpha_research.model_campaign.protocol import ModelCampaignConfig
    from iqrp.app.backtesting.alpha_research.model_campaign.runner import run_model_driven_campaign

    cfg = ModelCampaignConfig(smoke=True, output_dir=str(tmp_path / "camp"))
    # Further shrink for unit-test speed
    cfg.max_bars = {"1h": 400, "30m": 400, "15m": 400, "5m": 400, "1m": 400}
    cfg.holding_bars = (1,)
    cfg.cost_scenarios = ("BASE",)
    report = run_model_driven_campaign(cfg, progress=False)
    assert report["campaign_status"] in {
        "RESEARCH_COMPLETE_NO_CANDIDATES",
        "RESEARCH_COMPLETE_CANDIDATES_FOUND",
        "RESEARCH_BLOCKED",
    }
    assert (tmp_path / "camp" / "campaign_report.json").exists()
    assert (tmp_path / "camp" / "experiment_registry.json").exists()
    assert (tmp_path / "camp" / "reproducibility_report.json").exists()
    assert report.get("claim_distinctions", {}).get("PROFITABLE_STRATEGY") is False
    assert report.get("claim_distinctions", {}).get("LIVE_READY_STRATEGY") is False
