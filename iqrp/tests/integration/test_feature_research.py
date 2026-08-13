"""Integration + synthetic-data tests for feature research engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.features.research import FeatureResearchValidator, ResearchSettings


def _regime_frame(n: int = 300, seed: int = 11) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2023, 6, 1, tzinfo=UTC)
    # Two regimes: first half low vol, second high vol; predictive feature flips usefulness
    ret = np.concatenate([rng.normal(0.01, 0.2, n // 2), rng.normal(-0.01, 1.2, n - n // 2)])
    close = 100 + np.cumsum(ret)
    useful = np.r_[0.0, ret[:-1]]
    useless = rng.normal(size=n)
    twin = useful * 0.999 + rng.normal(0, 1e-6, size=n)
    rows = []
    for i in range(n):
        rows.append(
            {
                "open_time": start + timedelta(hours=i),
                "close": float(close[i]),
                "alpha": float(useful[i]),
                "beta_noise": float(useless[i]),
                "alpha_near": float(twin[i]),
                "open": float(close[i]),
                "high": float(close[i] + 1),
                "low": float(close[i] - 1),
                "volume": 1.0,
            }
        )
    return pl.DataFrame(rows)


@pytest.mark.integration
def test_synthetic_predictive_ranking(tmp_path: Path) -> None:
    frame = _regime_frame()
    settings = ResearchSettings.from_hydra(
        overrides=[
            "n_jobs=2",
            "predictive.min_train_size=80",
            "predictive.test_size=30",
            "predictive.step_size=30",
            "predictive.evaluation_mode=walk_forward",
            "importance.n_permutations=2",
            "importance.rfe_n_features_to_select=2",
            "importance.sfs_n_features_to_select=2",
            f"output_dir={tmp_path / 'reports'}",
            "drift.reference_fraction=0.4",
            "scoring.accept_score_threshold=40",
            "scoring.reject_score_threshold=10",
        ]
    )
    result = FeatureResearchValidator(settings).validate(
        frame, columns=["alpha", "beta_noise", "alpha_near"]
    )
    # Informative feature should show stronger predictive IC than pure noise
    assert abs(result.predictive["alpha"].mean_abs_ic) >= abs(
        result.predictive["beta_noise"].mean_abs_ic
    )
    assert result.correlation.high_correlation_groups or result.redundancy.near_duplicates
    assert any(result.drift[c].alerts is not None for c in result.drift)
    md = result.report_paths.get("markdown")
    assert md is not None and md.exists()


@pytest.mark.integration
def test_blocked_and_rolling_modes(tmp_path: Path) -> None:
    frame = _regime_frame(240)
    for mode in ("rolling", "blocked", "expanding"):
        settings = ResearchSettings.from_hydra(
            overrides=[
                f"predictive.evaluation_mode={mode}",
                "predictive.min_train_size=60",
                "predictive.test_size=20",
                "predictive.step_size=20",
                "predictive.blocked_n_splits=4",
                "importance.n_permutations=1",
                "importance.rfe_n_features_to_select=1",
                "importance.sfs_n_features_to_select=1",
                "reports.include_charts=false",
                f"output_dir={tmp_path / mode}",
            ]
        )
        result = FeatureResearchValidator(settings).validate(
            frame, columns=["alpha", "beta_noise"], write_reports=True
        )
        assert result.predictive["alpha"].by_target["future_return"].n_splits >= 1
