"""Focused tests for Prompt 40 candidate consolidation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.consolidation.protocol import (
    CORR_HIGHLY_REDUNDANT,
    CORR_RELATED,
    ConsolidationConfig,
)
from iqrp.app.backtesting.alpha_research.consolidation.runner import (
    _corr,
    behavioral_family_label,
    build_ensemble_positions,
    classify_redundancy,
    confidence_weights,
    connected_components,
    freq_bucket,
)


def test_thresholds_predeclared():
    assert CORR_HIGHLY_REDUNDANT == 0.85
    assert CORR_RELATED == 0.50
    cfg = ConsolidationConfig()
    assert "OOS" not in cfg.to_dict()["weighting_formula"] or "NO OOS" in cfg.to_dict()["weighting_formula"]


def test_redundancy_classes():
    cfg = ConsolidationConfig()
    assert classify_redundancy(0.9, cfg) == "HIGHLY_REDUNDANT"
    assert classify_redundancy(0.6, cfg) == "RELATED"
    assert classify_redundancy(0.2, cfg) == "DISTINCT"
    assert classify_redundancy(float("nan"), cfg) == "ANALYSIS_UNAVAILABLE"


def test_clustering_connected_components():
    ids = ["a", "b", "c", "d"]
    edges = {("a", "b"), ("c", "d")}
    comps = connected_components(ids, edges)
    assert len(comps) == 2
    assert sorted(comps[0]) in (["a", "b"], ["c", "d"])


def test_confidence_weights_validation_only_formula():
    w = confidence_weights([1.0, -2.0, 3.0], eps=0.05)
    assert abs(w.sum() - 1.0) < 1e-12
    assert w[1] < w[0] < w[2]  # negative sharpe gets only eps


def test_ensemble_methods():
    a = np.array([1, 1, -1, 0, 1], dtype=float)
    b = np.array([1, -1, -1, 0, 1], dtype=float)
    eq = build_ensemble_positions("equal_weight", [a, b])
    assert set(np.unique(eq)).issubset({-1.0, 0.0, 1.0})
    maj = build_ensemble_positions("majority_vote", [a, b, a])
    assert maj[0] == 1.0
    w = np.array([0.7, 0.3])
    cw = build_ensemble_positions("confidence_weighted", [a, b], weights=w)
    assert len(cw) == 5
    reg = np.array([1, 1, -1, 0, -1], dtype=float)
    rc = build_ensemble_positions("regime_conditioned", [a, b], regime_pos=reg)
    assert rc[3] == 0.0


def test_behavioral_family_not_model_name_alone():
    lab = behavioral_family_label(
        {"source_id": "momentum_signal:30m->5m", "family": "MTF", "kind": "mtf"}
    )
    assert "momentum" in lab.lower()
    assert "Multi-timeframe" in lab
    assert behavioral_family_label({"source_id": "cat_return_v1", "family": "CatBoost", "kind": "model"}) == (
        "ML directional family"
    )


def test_freq_buckets():
    cfg = ConsolidationConfig()
    assert freq_bucket(0.2, cfg) == "low-frequency"
    assert freq_bucket(3.0, cfg) == "moderate-frequency"
    assert freq_bucket(8.0, cfg) == "high-frequency"
    assert freq_bucket(25.0, cfg) == "overtrading-risk"


def test_corr_deterministic():
    rng = np.random.default_rng(40)
    x = rng.normal(size=100)
    y = x + rng.normal(size=100) * 0.1
    c1 = _corr(x, y)
    c2 = _corr(x, y)
    assert c1 == c2
    assert c1 > 0.8
