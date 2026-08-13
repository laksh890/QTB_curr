"""Unit tests for the feature research validation engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.core.exceptions import ConfigurationError, ValidationError
from iqrp.app.features.research import (
    FeatureResearchValidator,
    FeatureStatisticsEngine,
    ResearchSettings,
)
from iqrp.app.features.research._numeric import (
    binary_classification_metrics,
    distance_correlation,
    kendall,
    mutual_information,
    pearson,
    population_stability_index,
    ridge_fit_predict,
    roc_auc,
    spearman,
    try_mic,
)
from iqrp.app.features.research.cache import ResearchCache
from iqrp.app.features.research.correlation import CorrelationAnalyzer
from iqrp.app.features.research.drift import DriftDetector
from iqrp.app.features.research.importance import ImportanceAnalyzer
from iqrp.app.features.research.predictive_power import PredictivePowerEngine
from iqrp.app.features.research.redundancy import RedundancyDetector
from iqrp.app.features.research.reports import ReportWriter, ResearchReportDocument
from iqrp.app.features.research.stability import StabilityAnalyzer
from iqrp.app.features.research.targets import build_targets, select_feature_columns
from iqrp.app.features.research.timeseries_cv import iter_splits
from iqrp.app.features.research.visualization import ResearchVisualizer


def _window_mean(values: np.ndarray, idx: int, window: int) -> float:
    chunk = values[max(0, idx - window + 1) : idx + 1]
    finite = chunk[np.isfinite(chunk)]
    return float(np.mean(finite)) if finite.size else 0.0


def _synthetic(n: int = 180, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    # Latent return driver
    signal = np.cumsum(rng.normal(0, 0.2, size=n))
    noise = rng.normal(0, 1.0, size=n)
    close = 100 + np.cumsum(0.05 * signal + 0.2 * noise)
    feat_good = np.r_[np.nan, signal[:-1]]  # lagged signal → predicts next move
    feat_noise = rng.normal(0, 1, size=n)
    feat_dup = feat_good.copy()
    feat_const = np.ones(n)
    rows = []
    for i in range(n):
        rows.append(
            {
                "open_time": start + timedelta(minutes=i),
                "open": float(close[i]),
                "high": float(close[i] + 0.5),
                "low": float(close[i] - 0.5),
                "close": float(close[i]),
                "volume": float(10 + i % 5),
                "feat_good": float(feat_good[i]) if np.isfinite(feat_good[i]) else None,
                "feat_noise": float(feat_noise[i]),
                "feat_dup": float(feat_dup[i]) if np.isfinite(feat_dup[i]) else None,
                "feat_const": float(feat_const[i]),
                "feat_roll5": _window_mean(feat_good, i, 5),
                "feat_roll20": _window_mean(feat_good, i, 20),
            }
        )
    return pl.DataFrame(rows)


@pytest.mark.unit
def test_settings_from_hydra_and_overrides() -> None:
    settings = ResearchSettings.from_hydra(overrides=["scoring.accept_score_threshold=66"])
    assert settings.scoring.accept_score_threshold == 66
    assert ResearchSettings.default().enabled is True
    assert ResearchSettings.from_mapping({"n_jobs": 2}).n_jobs == 2


@pytest.mark.unit
def test_numeric_helpers() -> None:
    x = np.linspace(-1, 1, 80)
    y = x + np.random.default_rng(0).normal(0, 0.1, size=80)
    assert pearson(x, y) > 0.8
    assert spearman(x, y) > 0.8
    assert kendall(x, y) > 0.5
    assert distance_correlation(x, y) > 0.5
    assert mutual_information(x, y, bins=8) >= 0
    assert try_mic(x, y) is None or try_mic(x, y) >= 0
    pred = ridge_fit_predict(x[:60], y[:60], x[60:])
    assert len(pred) == 20
    metrics = binary_classification_metrics(y, x)
    assert "auc" in metrics
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) > 0.5
    psi = population_stability_index(x[:40], x[40:], bins=5)
    assert np.isfinite(psi)


@pytest.mark.unit
def test_timeseries_cv_modes() -> None:
    base = ResearchSettings.default()
    for mode in ("walk_forward", "expanding", "rolling", "blocked"):
        cfg = base.predictive.model_copy(update={"evaluation_mode": mode, "min_train_size": 40})
        splits = list(iter_splits(120, cfg))
        assert splits
        for s in splits:
            assert s.train_end <= s.test_start or mode == "blocked"


@pytest.mark.unit
def test_targets_and_statistics() -> None:
    frame = _synthetic(100)
    settings = ResearchSettings.default()
    cols = select_feature_columns(frame, settings)
    assert "feat_good" in cols
    targets = build_targets(frame, settings)
    assert "future_return" in targets.columns
    stats = FeatureStatisticsEngine(settings).compute(frame, cols)
    assert stats
    assert FeatureStatisticsEngine(settings).to_frame(stats).height == len(stats)
    with pytest.raises(ValidationError):
        build_targets(frame.drop("close"), settings)


@pytest.mark.unit
def test_correlation_redundancy_predictive() -> None:
    frame = _synthetic(160)
    settings = ResearchSettings.from_hydra(
        overrides=[
            "n_jobs=2",
            "predictive.min_train_size=50",
            "predictive.test_size=15",
            "predictive.step_size=15",
            "correlation.rolling_window=40",
        ]
    )
    cols = ["feat_good", "feat_noise", "feat_dup", "feat_const", "feat_roll5", "feat_roll20"]
    corr = CorrelationAnalyzer(settings).analyze(frame, cols)
    assert not corr.pearson.is_empty()
    assert corr.high_correlation_groups or corr.network_edges is not None
    red = RedundancyDetector(settings).detect(frame, cols)
    assert red.suggested_removals
    pred = PredictivePowerEngine(settings).evaluate(frame, cols)
    assert "feat_good" in pred
    assert pred["feat_good"].by_target["future_return"].n_splits >= 1


@pytest.mark.unit
def test_stability_drift_importance(tmp_path: Path) -> None:
    frame = _synthetic(200)
    settings = ResearchSettings.from_hydra(
        overrides=[
            "n_jobs=1",
            "importance.n_permutations=3",
            "importance.rfe_n_features_to_select=2",
            "importance.sfs_n_features_to_select=2",
            "importance.shap_enabled=true",
            f"cache_dir={tmp_path / 'cache'}",
            "cache_enabled=true",
        ]
    )
    cols = ["feat_good", "feat_noise", "feat_const"]
    stab = StabilityAnalyzer(settings).analyze(frame, cols)
    assert stab["feat_good"].stability_score >= 0
    drift = DriftDetector(settings).detect(frame, cols)
    assert "feat_good" in drift
    imp = ImportanceAnalyzer(settings).analyze(frame, cols)
    assert set(imp.permutation) == set(cols)
    assert imp.rfe_ranking
    assert imp.sfs_selected


@pytest.mark.unit
def test_cache_and_visualizer(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "c", enabled=True)
    frame = pl.DataFrame({"open_time": [datetime(2024, 1, 1, tzinfo=UTC)], "a": [1.0]})
    key = ResearchCache.make_key("t", frame, ["a"], {"x": 1})
    cache.put_frame(key, frame)
    assert cache.get_frame(key) is not None
    cache.put_json(key + "j", {"ok": True})
    assert cache.get_json(key + "j") == {"ok": True}
    assert ResearchCache(None, enabled=False).get_frame("x") is None

    viz = ResearchVisualizer(ResearchSettings.default())
    mat = pl.DataFrame(
        {"feature": ["a", "b"], "a": [1.0, 0.2], "b": [0.2, 1.0]},
    )
    paths = viz.write_all(
        tmp_path / "charts",
        corr_pearson=mat,
        rolling_ic={"a": [0.1, 0.2, 0.0, -0.1]},
        distributions={"a": np.random.default_rng(0).normal(size=100)},
        drift_psi={"a": 0.3, "b": 0.1},
        mi_ranking=[("a", 0.5), ("b", 0.1)],
        importance={"a": 0.2, "b": 0.01},
        stability={"a": 80.0, "b": 40.0},
    )
    assert paths
    assert paths["correlation_heatmap"].exists()


@pytest.mark.unit
def test_report_writer(tmp_path: Path) -> None:
    doc = ResearchReportDocument(
        summary={"n_features": 1},
        statistics=[{"name": "a", "mean": 0.0, "std": 1.0, "skewness": 0.0, "missing_pct": 0}],
        rankings={"top_features": [{"feature": "a", "score": 90}]},
        recommendations=["keep a"],
        accepted_features=[{"feature": "a", "score": 90, "reason": "ok"}],
        rejected_features=[],
        weak_features=[],
        correlated_groups=[["a", "b"]],
        charts={"x": "x.svg"},
        reasoning={"a": "ok"},
    )
    paths = ReportWriter().write(doc, tmp_path)
    assert paths["markdown"].exists()
    assert "Accepted" in paths["markdown"].read_text(encoding="utf-8")


@pytest.mark.unit
def test_validator_end_to_end(tmp_path: Path) -> None:
    frame = _synthetic(220)
    settings = ResearchSettings.from_hydra(
        overrides=[
            "n_jobs=2",
            "predictive.min_train_size=60",
            "predictive.test_size=20",
            "predictive.step_size=20",
            "importance.n_permutations=2",
            "importance.rfe_n_features_to_select=2",
            "importance.sfs_n_features_to_select=2",
            f"output_dir={tmp_path / 'out'}",
            f"cache_dir={tmp_path / 'cache'}",
            "visualization.max_features_in_charts=6",
        ]
    )
    result = FeatureResearchValidator(settings).validate(
        frame,
        columns=["feat_good", "feat_noise", "feat_dup", "feat_const", "feat_roll5"],
        asset_consistency={"feat_good": 80.0},
        timeframe_consistency={"feat_good": 75.0},
    )
    assert result.scores
    assert result.to_dict()["columns"]
    assert result.report_paths
    assert (tmp_path / "out" / "feature_research_report.md").exists()
    # Constant feature should not be accepted as a strong feature
    const_score = next(s for s in result.scores if s.feature == "feat_const")
    assert const_score.decision in {"reject", "weak"}


@pytest.mark.unit
def test_validator_disabled_and_empty() -> None:
    settings = ResearchSettings.from_mapping({"enabled": False})
    with pytest.raises(ConfigurationError):
        FeatureResearchValidator(settings).validate(_synthetic(50))
    settings2 = ResearchSettings.default()
    with pytest.raises(ValidationError):
        FeatureResearchValidator(settings2).validate(
            pl.DataFrame({"open_time": [datetime(2024, 1, 1, tzinfo=UTC)], "close": [1.0]})
        )


@pytest.mark.unit
def test_edge_branches_for_coverage(tmp_path: Path) -> None:
    settings = ResearchSettings.from_hydra(
        overrides=[
            "importance.shap_enabled=false",
            "importance.n_permutations=1",
            "importance.rfe_n_features_to_select=1",
            "importance.sfs_n_features_to_select=1",
            "visualization.enabled=false",
            "reports.include_charts=false",
            f"output_dir={tmp_path / 'out2'}",
            "columns.feature_prefix=feat_",
            "cache_enabled=false",
        ]
    )
    frame = _synthetic(120)
    # empty feature stats / empty corr
    assert CorrelationAnalyzer(settings).analyze(frame, []).pearson.is_empty()
    assert RedundancyDetector(settings).detect(frame, []).feature_count == 0
    empty_imp = ImportanceAnalyzer(settings).analyze(frame, [])
    assert empty_imp.rfe_ranking == []

    # prefix filtering + shap disabled path + no charts
    result = FeatureResearchValidator(settings).validate(frame, write_reports=True)
    assert all(c.startswith("feat_") for c in result.columns)
    assert result.accepted() or result.weak() or result.rejected()
    _ = result.accepted(), result.weak(), result.rejected()

    # numeric edge helpers
    from iqrp.app.features.research._numeric import (
        as_float_matrix,
        clip01,
        ks_statistic,
        r_squared,
        shannon_entropy,
    )

    assert as_float_matrix([]).shape == (0, 0)
    assert clip01(float("nan")) == 0.0
    assert clip01(2.0) == 1.0
    assert np.isnan(r_squared(np.array([1.0]), np.array([1.0])))
    assert np.isnan(shannon_entropy(np.array([1.0, 2.0]), bins=20))
    assert np.isnan(ks_statistic(np.array([1.0]), np.array([2.0])))

    # empty / constant stats
    stats = FeatureStatisticsEngine(settings).compute(
        pl.DataFrame({"x": [None, None], "y": [1.0, 1.0]}),
        ["x", "y"],
    )
    assert stats[0].distribution_type in {"empty", "insufficient_data"}
    assert stats[1].distribution_type in {"constant", "insufficient_data"}

    # cache disabled paths
    cache = ResearchCache(tmp_path / "c2", enabled=False)
    assert cache.get_frame("k") is None
    cache.put_frame("k", frame)
    cache.put_json("k", {"a": 1})
    assert cache.get_json("k") is None

    # config invalid path
    from omegaconf import OmegaConf

    with pytest.raises(ConfigurationError):
        ResearchSettings.from_mapping(OmegaConf.create([1, 2, 3]))
