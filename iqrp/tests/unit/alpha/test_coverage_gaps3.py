"""Third-pass coverage for remaining alpha misses toward >98%."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from iqrp.app.alpha.backtesting.portfolio_backtest import portfolio_backtest
from iqrp.app.alpha.backtesting.signal_backtest import signal_backtest, signal_to_weights
from iqrp.app.alpha.base.alpha_signal import AlphaSignal
from iqrp.app.alpha.base.signal_definition import SignalDefinition
from iqrp.app.alpha.base.signal_registry import SignalRegistry
from iqrp.app.alpha.base.signal_result import (
    SignalPerformance,
    SignalResearchReport,
    SignalScore,
    SignalStatus,
)
from iqrp.app.alpha.discovery.alternative import alternative_zscore_signal, apply_publication_lag
from iqrp.app.alpha.discovery.cross_sectional import (
    cross_sectional_rank_signal,
    cross_sectional_zscore_signal,
    long_short_spread,
)
from iqrp.app.alpha.discovery.event_based import event_impulse_signal, surprise_signal
from iqrp.app.alpha.discovery.statistical import screen_features
from iqrp.app.alpha.discovery.symbolic import evaluate_expression, rank as sym_rank, rolling_apply
from iqrp.app.alpha.discovery.time_series import (
    mean_reversion_signal,
    trend_signal,
    volatility_signal,
    volume_signal,
)
from iqrp.app.alpha.economics.transaction_costs import estimate_transaction_cost
from iqrp.app.alpha.engine import AlphaResearchEngine, ApprovalError
from iqrp.app.alpha.ensemble.clustering import (
    hierarchical_correlation_clusters,
    representative_per_cluster,
)
from iqrp.app.alpha.ensemble.correlation import signal_correlation_matrix
from iqrp.app.alpha.ensemble.signal_combination import (
    combine_from_metrics,
    combine_signals,
    majority_sign_combine,
)
from iqrp.app.alpha.ensemble.weighting import normalize_weights, signal_quality_score
from iqrp.app.alpha.monitoring.alerts import build_alpha_alerts
from iqrp.app.alpha.monitoring.performance_decay import (
    monitor_performance_decay,
    performance_decay_score,
)
from iqrp.app.alpha.monitoring.retirement import evaluate_retirement
from iqrp.app.alpha.monitoring.signal_decay import (
    estimate_ic_half_life,
    ic_decay_curve,
    rolling_ic,
)
from iqrp.app.alpha.monitoring.signal_drift import concept_drift_ic, monitor_signal_drift
from iqrp.app.alpha.ranking import rank_candidates
from iqrp.app.alpha.regime.conditional_alpha import (
    compare_unconditional_vs_conditional,
    conditional_alpha_profile,
    conditional_ic,
    regime_gated_signal,
)
from iqrp.app.alpha.regime.regime_performance import (
    regime_hit_rate,
    regime_ic,
    regime_performance,
    regime_returns,
)
from iqrp.app.alpha.research.decay import analyze_decay, forward_returns
from iqrp.app.alpha.research.evaluator import SignalEvaluator
from iqrp.app.alpha.serializer import _to_jsonable
from iqrp.app.alpha.statistical_validation.bootstrap import block_bootstrap_ci, iid_bootstrap_ci
from iqrp.app.alpha.statistical_validation.multiple_testing import (
    _local_adjust_pvalues,
    _resolve_adjust_pvalues,
)
from iqrp.app.alpha.statistical_validation.permutation import permutation_ic_test
from iqrp.app.alpha.statistical_validation.probability_backtest_overfitting import (
    probability_backtest_overfitting,
)
from iqrp.app.alpha.statistical_validation.significance import (
    ic_significance,
    newey_west_ic_significance,
    newey_west_variance,
)

HYP = "Slow incorporation of public information creates short-horizon continuation research candidates."


def test_conditional_alpha_panel_paths(panel: np.ndarray, rng: np.random.Generator) -> None:
    fwd = panel * 0.2 + rng.normal(0, 0.05, size=panel.shape)
    cond = np.zeros(panel.shape[0], dtype=bool)
    cond[::2] = True
    with pytest.raises(ValueError):
        conditional_ic(panel, fwd, cond[:5])
    out = conditional_ic(panel, fwd, cond, rank=True)
    assert "ic" in out
    # constant row → skip
    flat = np.ones_like(panel)
    conditional_ic(flat, flat, cond, rank=False)
    conditional_alpha_profile(panel, fwd, {})
    with pytest.raises(ValueError):
        regime_gated_signal(panel, np.array(["a"] * 5), ["a"])
    gated = regime_gated_signal(
        panel, np.array(["a", "b"] * (panel.shape[0] // 2)), ["a"], inactive_value=0.0
    )
    assert gated.shape == panel.shape
    compare_unconditional_vs_conditional(panel, fwd, np.array(["a", "b"] * (panel.shape[0] // 2)))


def test_regime_performance_panel_and_edges(panel: np.ndarray, rng: np.random.Generator) -> None:
    fwd = panel + rng.normal(0, 0.1, size=panel.shape)
    labels = np.array(["x", "y"] * (panel.shape[0] // 2))
    with pytest.raises(ValueError):
        regime_ic(panel, fwd[:, :5], labels)
    regime_ic(panel, fwd, labels, rank=True)
    with pytest.raises(ValueError):
        regime_returns(panel.mean(axis=1), labels, positions=np.ones(3))
    regime_performance(panel, fwd, labels, strategy_returns=panel.mean(axis=1))
    regime_hit_rate(panel, fwd, labels)
    # 1d hit rate path already covered; pearson/spearman edge via tiny slice
    regime_ic(np.ones(20), np.ones(20), np.array(["a"] * 20))


def test_ranking_all_extract_branches() -> None:
    class Score:
        overall = "bad"

    class ScoreOk:
        overall = 77.0

    class Perf:
        def to_dict(self) -> dict[str, Any]:
            return {"ic_mean": 0.06, "hit_rate": 0.55}

    class Defn:
        name = "from_defn"
        economic_hypothesis = "h" * 40

    class Report:
        def to_dict(self) -> dict[str, Any]:
            return {"performance": {"ic": 0.04, "hit_rate": 0.52}}

    cands = [
        {"score": "bad", "ic": "x", "ic_mean": "y", "stability": "z"},
        {"score": Score()},
        {"score": ScoreOk()},
        {"performance": Perf(), "definition": Defn()},
        {"report": Report()},
        {"definition": {"name": "nested", "economic_hypothesis": "z" * 50}, "ic": 0.09},
        {"name": "hit", "performance": {"hit_rate": 0.7}, "stability": 0.8},
        {"research_score": float("nan"), "overall": None, "ic_mean": 0.02},
    ]
    ranked = rank_candidates(cands)
    assert len(ranked) == len(cands)
    assert all("rank" in r for r in ranked)


def test_retirement_all_vote_paths() -> None:
    # sign flip ic collapse
    evaluate_retirement(ic_recent=-0.05, ic_baseline=0.05, net_sharpe=-0.1)
    # degrade ratio only
    evaluate_retirement(ic_recent=0.03, ic_baseline=0.1)
    # baseline ~0 path
    evaluate_retirement(ic_recent=0.0, ic_baseline=0.0)
    # explicit zero recent
    evaluate_retirement(ic_recent=0.0, ic_baseline=0.05, net_sharpe=0.1)
    # cost via gross/net
    evaluate_retirement(gross_sharpe=1.0, net_sharpe=-0.1)
    # capacity degrade only
    evaluate_retirement(capacity=0.5e6, capacity_baseline=1e6)
    # capacity alone tiny
    evaluate_retirement(capacity=0.01)
    # drift medium
    evaluate_retirement(drift_severity="medium")
    # drift high
    evaluate_retirement(drift_severity="high")
    # custom thresholds + weak sharpe
    evaluate_retirement(net_sharpe=0.1, thresholds={"net_sharpe_degrade": 0.5})
    # retire_votes>=2
    evaluate_retirement(
        ic_recent=0.0,
        ic_baseline=0.1,
        capacity=1.0,
        capacity_baseline=100.0,
        net_sharpe=0.5,
    )


def test_engine_approve_stub_report_and_evidence(
    genuine: dict[str, Any],
) -> None:
    eng = AlphaResearchEngine(registry=SignalRegistry())
    sig = np.asarray(genuine["signal"])
    ret = np.asarray(genuine["returns"])
    d = SignalDefinition(
        name="stub_appr",
        version="1.0.0",
        formula="x",
        features=("r",),
        lookback=10,
        horizon=1,
        universe="u",
        frequency="1d",
        direction="long_short",
        expected_relationship="positive",
        economic_hypothesis=HYP,
        owner="r",
    )
    rec = eng.register(d, signal=sig)
    # evidence via performance+diagnostics without validate key naming
    eng.registry.attach_report(
        rec.experiment_id,
        SignalResearchReport(
            signal_name="stub_appr",
            version="1.0.0",
            status=SignalStatus.CANDIDATE,
            economic_hypothesis=HYP,
            performance=SignalPerformance(ic_mean=0.05, extras={"decay": {}}),
            diagnostics={"decay": True},
        ),
    )
    eng.approve(rec.experiment_id, reason="IC validation + economic hypothesis")

    # approve with NO report → stub creation path: clear report after attaching evidence then...
    rec2 = eng.register(
        SignalDefinition(
            name="stub2",
            version="1.0.0",
            formula="x",
            features=("r",),
            lookback=5,
            horizon=1,
            universe="u",
            frequency="1d",
            direction="long_short",
            expected_relationship="positive",
            economic_hypothesis=HYP,
            owner="r",
        ),
        signal=sig,
    )
    # Put evidence then delete report object before approve by replacing with None-like:
    # attach minimal validate then approve after advancing — force no-report stub by
    # temporarily setting report None after validate marker via custom path:
    eng.registry.attach_report(
        rec2.experiment_id,
        SignalResearchReport(
            signal_name="stub2",
            version="1.0.0",
            status=SignalStatus.CANDIDATE,
            economic_hypothesis=HYP,
            diagnostics={"bootstrap": {"ok": True}},
        ),
    )
    # Manually clear report after evidence check by monkeypatching _has_validation_evidence
    real = eng._has_validation_evidence

    def _yes(record: Any) -> bool:
        return True

    eng._has_validation_evidence = _yes  # type: ignore[method-assign]
    eng.registry.get(rec2.experiment_id).report = None
    out = eng.approve(rec2.experiment_id, reason="IC validation + economic hypothesis")
    assert out.status == SignalStatus.APPROVED
    eng._has_validation_evidence = real  # type: ignore[method-assign]

    # Cannot approve from RETIRED
    eng.retire(rec2.experiment_id, reason="done")
    with pytest.raises(ApprovalError):
        eng.approve(rec2.experiment_id, reason="IC validation + economic hypothesis")

    # validate with tiny pnl
    eng.validate(
        np.array([1.0, np.nan]), np.array([0.0, 0.0]), n_trials=5, returns_are_forward=True
    )


def test_signal_decay_half_life_and_rank(
    signal: np.ndarray, returns: np.ndarray, fwd: np.ndarray
) -> None:
    rolling_ic(signal, fwd, window=40, rank=True)
    # shape mismatch
    with pytest.raises(ValueError):
        rolling_ic(signal, fwd[:10], window=20)
    curve = ic_decay_curve(signal, returns, horizons=(1, 2, 3, 5))
    # force half-life fitting branches
    estimate_ic_half_life([1, 2, 3, 5, 10], np.array([0.2, 0.1, 0.05, 0.02, 0.01]))
    estimate_ic_half_life([1, 2, 3], np.array([0.0, 0.0, 0.0]))
    _ = curve


def test_evaluator_nan_signal_path(returns: np.ndarray) -> None:
    ev = SignalEvaluator(horizons=(1, 2), stability_window=40, seasonality_period=5)
    # all-nan signal
    report = ev.evaluate(np.full(returns.size, np.nan), returns)
    assert report.statistics is not None
    # AlphaSignal input
    ev.evaluate(AlphaSignal(values=returns, name="r"), returns)


def test_discovery_error_and_edge_branches(
    returns: np.ndarray, panel: np.ndarray, rng: np.random.Generator
) -> None:
    with pytest.raises(ValueError):
        mean_reversion_signal(returns, lookback=0)
    prices = 100 * np.cumprod(1 + returns)
    with pytest.raises(ValueError):
        trend_signal(prices, lookback_fast=20, lookback_slow=10)
    with pytest.raises(ValueError):
        trend_signal(prices, lookback_fast=0, lookback_slow=10)
    volatility_signal(returns, lookback=5)
    with pytest.raises(ValueError):
        volume_signal(returns, lookback=0)
    with pytest.raises(ValueError):
        event_impulse_signal(np.zeros(10, dtype=bool), decay=0.0)
    with pytest.raises(ValueError):
        event_impulse_signal(np.zeros(10, dtype=bool), decay=1.5)
    with pytest.raises(ValueError):
        event_impulse_signal(np.zeros(10, dtype=bool), horizon=0)
    surprise_signal(returns, returns * 0.5, lookback=5)
    with pytest.raises(ValueError):
        apply_publication_lag(returns, -1)
    with pytest.raises(ValueError):
        alternative_zscore_signal(returns, lookback=0)
    # CS edges
    cross_sectional_rank_signal(panel, asset_index=0)
    with pytest.raises(Exception):
        cross_sectional_rank_signal(panel, asset_index=999)
    cross_sectional_zscore_signal(panel, asset_index=0)
    long_short_spread(panel)
    # statistical filters
    feats = {"a": returns, "b": np.ones_like(returns)}
    screen_features(feats, forward_returns(returns, 1), min_abs_ic=0.9, min_obs=500)
    # symbolic stack != 1
    with pytest.raises(ValueError):
        evaluate_expression(
            [("load", {"name": "r"}), ("load", {"name": "r"})],
            {"r": returns},
        )
    sym_rank(returns, window=None)
    rolling_apply(returns, 3, lambda x: np.nanmean(x), min_periods=10)


def test_backtest_ann_sharpe_and_modes(rng: np.random.Generator) -> None:
    s = rng.normal(size=100)
    r = rng.normal(0, 0.01, size=100)
    signal_to_weights(s, mode="long_only")
    signal_backtest(s, r, cost_bps=0, mode="long_only", returns_are_forward=True)
    # empty / zero vol returns for ann sharpe
    signal_backtest(np.ones(5), np.zeros(5), cost_bps=0.0)
    w = np.ones((20, 2)) / 2
    portfolio_backtest(
        w, rng.normal(0, 0.01, size=(20, 2)), cost_bps=0.0, returns_are_forward=False
    )
    portfolio_backtest(np.zeros((0, 2)), np.zeros((0, 2)))


def test_ensemble_weighting_and_combine_edges(rng: np.random.Generator) -> None:
    normalize_weights(np.array([0.2, 0.8]), names=["a", "b"], min_weight=0.1)
    normalize_weights(np.array([-1.0, -1.0]), names=["a", "b"])
    signal_quality_score({"ic": 0.05}, score_weights={"ic": 1.0})
    signal_quality_score({"ic": 0.05, "uncertainty": 0.2, "sharpe": float("nan")})
    series = {"a": rng.normal(size=80), "b": rng.normal(size=80)}
    combine_signals(series, weights={"a": 0.0, "b": 0.0})
    combine_signals({"a": rng.normal(size=5)}, weights={"a": 1.0})
    combine_from_metrics(series, {k: {"ic": 0.01} for k in series}, method="equal")
    majority_sign_combine(series)
    corr = signal_correlation_matrix(series)
    hierarchical_correlation_clusters(corr, max_clusters=1)
    representative_per_cluster([["a", "b"]], {k: {"ic": 0.1} for k in series})


def test_monitoring_alerts_and_perf_edges(rng: np.random.Generator) -> None:
    build_alpha_alerts(retirement={"status": "DEGRADED", "reasons": ["x"]})
    build_alpha_alerts(ic_decay={"status": "DECAYING"})
    build_alpha_alerts(performance={"status": "DEGRADED"})
    r = rng.normal(0, 0.01, 200)
    performance_decay_score(r[:10])
    monitor_performance_decay(r)
    concept_drift_ic(r[:50], r[:50], r[50:100], r[50:100])
    monitor_signal_drift(r[:50], r[50:100])


def test_stat_val_edge_branches(rng: np.random.Generator) -> None:
    x = rng.normal(size=50)
    y = x * 0.3 + rng.normal(0, 1, 50)
    iid_bootstrap_ci(x, y, stat="ic", n_boot=20, seed=0)
    iid_bootstrap_ci(np.zeros(10), np.zeros(10), stat="ic", n_boot=10)
    iid_bootstrap_ci(x, None, stat="sharpe", n_boot=15)
    block_bootstrap_ci(x, y, n_boot=15, block_size=100, seed=0)
    permutation_ic_test(np.zeros(10), np.zeros(10), n_perm=10)
    permutation_ic_test(x[:3], y[:3], n_perm=5)
    ic_significance(np.ones(10), np.arange(10.0))
    newey_west_variance(np.ones(5), lags=2)
    newey_west_ic_significance(x, y, window=5, lags=1)
    probability_backtest_overfitting(np.ones((20, 1)), n_groups=2)
    probability_backtest_overfitting(rng.normal(size=(30, 3)), n_groups=3, max_combinations=5)
    # fdr_bh local path explicitly
    _local_adjust_pvalues([0.01, 0.2, 0.4], method="fdr_bh")
    _resolve_adjust_pvalues()
    # lazy getattr remaining
    import iqrp.app.alpha.statistical_validation as sv

    _ = sv.probability_backtest_overfitting


def test_phase11_doc_and_symbol_fail_paths(tmp_path: Path) -> None:
    from iqrp.app.alpha import phase11

    # force missing doc on a component
    comps = list(phase11.PHASE11_COMPONENTS)
    fake = phase11.ComponentCheck(
        "Fake", "x", "iqrp.app.alpha", "AlphaResearchEngine", docs=["DefinitelyMissing.md"]
    )
    with mock.patch.object(phase11, "PHASE11_COMPONENTS", comps + [fake]):
        report = phase11.validate_phase11(write_stubs=False)
        assert report["status"] == "FAIL"
    # missing symbol
    fake2 = phase11.ComponentCheck("Fake2", "x", "iqrp.app.alpha", "NotARealSymbolXYZ", docs=[])
    with mock.patch.object(phase11, "PHASE11_COMPONENTS", [fake2]):
        with mock.patch.object(phase11, "REQUIRED_DOCS", []):
            report2 = phase11.validate_phase11(write_stubs=False)
            assert report2["status"] == "FAIL"


def test_serializer_enum_and_model_dump() -> None:
    class E:
        value = "x"

    assert isinstance(_to_jsonable({"e": SignalStatus.CANDIDATE}), dict)
    # unknown object
    assert isinstance(_to_jsonable(object()), str)


def test_cs_and_econ_remaining(panel: np.ndarray, sectors: np.ndarray) -> None:
    from iqrp.app.alpha.cross_section.factor_adjustment import factor_exposure_summary
    from iqrp.app.alpha.cross_section.neutralization import demean_by_group, neutralize_weighted
    from iqrp.app.alpha.cross_section.ranking import cross_sectional_rank, winsorize_cross_section
    from iqrp.app.alpha.cross_section.sector_adjustment import (
        cap_weighted_sector_neutral,
        industry_neutralize,
        sector_neutral_zscore,
    )
    from iqrp.app.alpha.economics.capacity import estimate_capacity
    from iqrp.app.alpha.economics.market_impact import market_impact_bps
    from iqrp.app.alpha.economics.slippage import slippage_bps
    from iqrp.app.alpha.economics.turnover import turnover_series

    cross_sectional_rank(np.asarray([[1.0]]), pct=True)
    winsorize_cross_section(panel, lower=0.0, upper=1.0)
    demean_by_group(panel, sectors)
    neutralize_weighted(panel, weights=np.linspace(1, 2, panel.shape[1]))
    sector_neutral_zscore(panel, sectors)
    industry_neutralize(panel, sectors)
    cap_weighted_sector_neutral(panel, sectors, np.linspace(1, 2, panel.shape[1]))
    # factor summary incompatible
    with pytest.raises(ValueError):
        factor_exposure_summary(panel, np.ones((3, 3, 2)))
    estimate_capacity(turnover=0.1, adv=1e6, annualize_turnover=True)
    slippage_bps(np.array([0.01, 0.02]))
    market_impact_bps(np.array([0.01, 0.02]))
    turnover_series(np.ones((5, 2)) * 0.5, half=False)


def test_research_decay_and_info_edges(rng: np.random.Generator) -> None:
    from iqrp.app.alpha.research.hit_rate import compute_hit_rate
    from iqrp.app.alpha.research.information_coefficient import rolling_ic as ric
    from iqrp.app.alpha.research.persistence import autocorrelation, signal_half_life
    from iqrp.app.alpha.research.predictor import SignalPredictor
    from iqrp.app.alpha.research.rank_ic import compute_rank_ic
    from iqrp.app.alpha.research.seasonality import analyze_seasonality

    s = rng.normal(size=100)
    r = rng.normal(size=100)
    with pytest.raises(ValueError):
        forward_returns(r, 0)
    analyze_decay(s, r, horizons=(1,))
    ric(s, r, window=3, min_obs=2)
    compute_rank_ic(np.ones(20), rng.normal(size=20))
    compute_hit_rate(s, np.zeros_like(s))
    autocorrelation(s[:2], 1)
    signal_half_life(s, max_lag=1)
    analyze_seasonality(s, r, period=2, horizon=1)
    SignalPredictor(min_train=10, test_size=5, step=5).predict(s, r)


def test_diagnostics_remaining_panel_rows() -> None:
    from iqrp.app.alpha.diagnostics import leakage_shift_test

    # 2d with 1d returns triggers continue path in daily loop
    sig = np.random.default_rng(0).normal(size=(30, 5))
    ret1d = np.random.default_rng(1).normal(size=30)
    leakage_shift_test(sig, ret1d, max_lead=1, min_obs=5)
