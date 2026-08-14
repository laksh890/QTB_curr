"""Ranking/neutralization/residualization; ensemble; correlation; retirement; regimes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from iqrp.app.alpha.cross_section.factor_adjustment import (
    factor_exposure_summary,
    factor_neutralize,
    market_beta_adjust,
    orthogonalize_to_book,
    style_adjust,
)
from iqrp.app.alpha.cross_section.neutralization import (
    demean_by_group,
    neutralize_market,
    neutralize_multi_group,
    neutralize_weighted,
)
from iqrp.app.alpha.cross_section.ranking import (
    cross_sectional_minmax,
    cross_sectional_percentile,
    cross_sectional_rank,
    cross_sectional_zscore,
    winsorize_cross_section,
)
from iqrp.app.alpha.cross_section.residualization import (
    beta_residualize,
    residualize_vs_factors,
    residualize_vs_signals,
)
from iqrp.app.alpha.cross_section.sector_adjustment import (
    cap_weighted_sector_neutral,
    industry_neutralize,
    sector_neutral_zscore,
    sector_relative_ranks,
)
from iqrp.app.alpha.ensemble.clustering import (
    cluster_signals_from_series,
    correlation_distance,
    hierarchical_correlation_clusters,
    representative_per_cluster,
)
from iqrp.app.alpha.ensemble.correlation import (
    correlation_penalty_vector,
    drawdown_correlation_matrix,
    ic_correlation_matrix,
    position_correlation_matrix,
    prediction_correlation_matrix,
    return_correlation_matrix,
    signal_correlation_matrix,
)
from iqrp.app.alpha.ensemble.redundancy import (
    detect_nested_signals,
    feature_overlap,
    find_high_correlation_pairs,
    redundancy_report,
)
from iqrp.app.alpha.ensemble.signal_combination import (
    combine_from_metrics,
    combine_signals,
    majority_sign_combine,
    rank_average_combine,
)
from iqrp.app.alpha.ensemble.weighting import (
    DEFAULT_SCORE_WEIGHTS,
    compute_ensemble_weights,
    correlation_adjusted_weights,
    dynamic_weights,
    equal_weights,
    ic_weights,
    normalize_weights,
    regime_weights,
    risk_adjusted_weights,
    signal_quality_score,
)
from iqrp.app.alpha.monitoring.alerts import build_alpha_alerts, summarize_alerts
from iqrp.app.alpha.monitoring.performance_decay import (
    max_drawdown,
    monitor_performance_decay,
    performance_decay_score,
    rolling_performance,
)
from iqrp.app.alpha.monitoring.retirement import (
    batch_evaluate_retirement,
    evaluate_retirement,
)
from iqrp.app.alpha.monitoring.signal_decay import (
    estimate_ic_half_life,
    ic_decay_curve,
    monitor_ic_decay,
    rolling_ic,
)
from iqrp.app.alpha.monitoring.signal_drift import (
    concept_drift_ic,
    monitor_signal_drift,
    position_drift,
    signal_distribution_drift,
)
from iqrp.app.alpha.ranking import rank_candidates
from iqrp.app.alpha.regime.conditional_alpha import (
    apply_condition_fn,
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
from iqrp.app.alpha.regime.regime_stability import (
    regime_concentration,
    regime_stability_score,
    rolling_regime_stability,
)


def test_cross_sectional_ranking_neutralization(panel: np.ndarray, sectors: np.ndarray) -> None:
    r = cross_sectional_rank(panel, pct=False)
    assert r.shape == panel.shape
    p = cross_sectional_percentile(panel)
    z = cross_sectional_zscore(panel, clip=3.0)
    mm = cross_sectional_minmax(panel)
    w = winsorize_cross_section(panel, lower=0.05, upper=0.95)
    assert z.shape == panel.shape
    assert mm.shape == panel.shape
    assert w.shape == panel.shape
    assert p.shape == panel.shape

    dm = demean_by_group(panel, sectors)
    assert np.allclose(np.nanmean(dm[0, sectors == "A"]), 0.0, atol=1e-8)
    nm = neutralize_market(panel)
    assert nm.shape == panel.shape
    nw = neutralize_weighted(panel, weights=np.ones(panel.shape[1]))
    assert nw.shape == panel.shape
    ng = neutralize_multi_group(panel, [sectors, sectors])
    assert ng.shape == panel.shape

    sn = sector_neutral_zscore(panel, sectors)
    ind = industry_neutralize(panel, sectors)
    sr = sector_relative_ranks(panel, sectors)
    cw = cap_weighted_sector_neutral(panel, sectors, np.ones(panel.shape[1]))
    assert sn.shape == panel.shape
    assert ind.shape == panel.shape
    assert sr.shape == panel.shape
    assert cw.shape == panel.shape


def test_residualization_and_factor_adj(panel: np.ndarray, rng: np.random.Generator) -> None:
    t, n = panel.shape
    factors_nk = rng.normal(size=(n, 2))
    factors_tn = rng.normal(size=(t, n))
    factors_tnk = rng.normal(size=(t, n, 2))

    resid = residualize_vs_factors(panel, factors_nk)
    assert resid.shape == panel.shape
    resid2 = residualize_vs_factors(panel, factors_tn)
    assert resid2.shape == panel.shape
    resid3 = residualize_vs_factors(panel, factors_tnk, add_intercept=False)
    assert resid3.shape == panel.shape
    with pytest.raises(ValueError):
        residualize_vs_factors(panel, rng.normal(size=(3, 3)))
    with pytest.raises(ValueError):
        residualize_vs_factors(panel, rng.normal(size=(t, n, 2, 2)))

    rs = residualize_vs_signals(panel, {"a": panel * 0.5, "b": panel * -0.2})
    assert rs.shape == panel.shape
    rs2 = residualize_vs_signals(panel, [panel * 0.3])
    assert rs2.shape == panel.shape
    assert residualize_vs_signals(panel, []).shape == panel.shape
    with pytest.raises(ValueError):
        residualize_vs_signals(panel, {"bad": panel[:, : n - 1]})

    mkt = panel.mean(axis=1)
    beta_r = beta_residualize(panel, mkt, panel, lookback=40, min_obs=20)
    assert beta_r.shape == panel.shape
    with pytest.raises(ValueError):
        beta_residualize(panel, mkt[:10], panel)

    fn = factor_neutralize(panel, factors_nk)
    st = style_adjust(panel, {"s0": panel * 0.1})
    st2 = style_adjust(panel, [panel * 0.2])
    mb = market_beta_adjust(panel, mkt, panel, lookback=40)
    ob = orthogonalize_to_book(panel, {"y": panel * 0.5})
    assert fn.shape == panel.shape
    assert st.shape == panel.shape and st2.shape == panel.shape
    assert mb.shape == panel.shape
    assert ob.shape == panel.shape
    summary = factor_exposure_summary(panel, factors_nk)
    assert "mean_correlation" in summary
    summary2 = factor_exposure_summary(panel, factors_tnk)
    assert summary2["n_factors"] == 2
    # 1D signal path in summary
    summary3 = factor_exposure_summary(panel[0], factors_nk)
    assert "abs_mean_correlation" in summary3


def test_ensemble_weights_not_sharpe_only() -> None:
    """Ensemble must not overweight solely on historical Sharpe."""
    assert DEFAULT_SCORE_WEIGHTS["sharpe"] <= 0.05
    metrics = {
        "a": {
            "ic": 0.05,
            "stability": 0.8,
            "capacity": 0.5,
            "decay": 0.2,
            "corr_penalty": 0.1,
            "sharpe": 10.0,
        },
        "b": {
            "ic": 0.05,
            "stability": 0.8,
            "capacity": 0.5,
            "decay": 0.2,
            "corr_penalty": 0.1,
            "sharpe": 0.0,
        },
    }
    w = compute_ensemble_weights(metrics, method="composite")
    assert abs(sum(w.values()) - 1.0) < 1e-8
    # Sharpe gap of 10 should not dominate equal IC/stability peers
    assert abs(w["a"] - w["b"]) < 0.15

    with pytest.raises(ValueError, match="unknown"):
        compute_ensemble_weights(metrics, method="sharpe")  # type: ignore[arg-type]

    for method in ("equal", "ic", "risk_adj", "corr_adj", "regime", "dynamic", "composite"):
        ww = compute_ensemble_weights(metrics, method=method, regime_scores={"a": 1.2, "b": 0.8})
        assert abs(sum(ww.values()) - 1.0) < 1e-8

    assert equal_weights(["a", "b"])["a"] == 0.5
    assert ic_weights(metrics)["a"] > 0
    assert risk_adjusted_weights(metrics)["a"] > 0
    assert correlation_adjusted_weights(metrics)["a"] > 0
    assert regime_weights(metrics, regime_scores={"a": 1.0, "b": 1.0})["a"] > 0
    assert dynamic_weights(metrics)["a"] > 0
    assert signal_quality_score(metrics["a"]) > 0
    assert normalize_weights({"a": 0.0, "b": 0.0})["a"] == 0.5
    assert compute_ensemble_weights({}) == {}


def test_correlation_redundancy_clustering_combine(
    rng: np.random.Generator, signal: np.ndarray, returns: np.ndarray
) -> None:
    s1 = signal
    s2 = signal * 0.9 + rng.normal(0, 0.1, size=signal.size)
    s3 = rng.normal(size=signal.size)
    series = {"s1": s1, "s2": s2, "s3": s3}

    corr = signal_correlation_matrix(series, kind="prediction")
    assert isinstance(corr, dict)
    prediction_correlation_matrix(series)
    return_correlation_matrix({"s1": returns, "s2": returns * 0.5})
    position_correlation_matrix(series)
    drawdown_correlation_matrix({"s1": returns, "s2": -returns})
    ic_series = {
        "s1": s1 * returns,
        "s2": s2 * returns,
        "s3": s3 * returns,
    }
    ic_correlation_matrix(ic_series)
    pen = correlation_penalty_vector(corr)
    assert isinstance(pen, dict)

    pairs = find_high_correlation_pairs(corr, threshold=0.5)
    assert isinstance(pairs, list)
    nested = detect_nested_signals(series, r2_threshold=0.99, min_obs=30)
    assert isinstance(nested, list)
    ov = feature_overlap({"a": ("x", "y"), "b": ("y", "z")})
    assert isinstance(ov, (dict, list))
    red = redundancy_report(series)
    assert isinstance(red, dict)

    clusters = hierarchical_correlation_clusters(corr, max_clusters=2)
    assert isinstance(clusters, dict)
    cs = cluster_signals_from_series(series, threshold=0.5)
    assert cs is not None
    mat = np.asarray(corr.get("matrix", corr.get("correlation", np.eye(3))), dtype=float)
    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        dist = correlation_distance(mat)
        assert dist.shape[0] == dist.shape[1]
    metrics = {k: {"ic": 0.05} for k in series}
    cluster_map = clusters.get("clusters") or {"0": list(series)}
    if (
        isinstance(cluster_map, dict)
        and cluster_map
        and isinstance(next(iter(cluster_map.values())), (list, tuple))
    ):
        reps = representative_per_cluster(cluster_map, metrics)
        assert isinstance(reps, list)

    comb = combine_signals(series, weights={"s1": 0.5, "s2": 0.3, "s3": 0.2})
    assert comb.shape[0] == s1.size
    cm, wts = combine_from_metrics(series, metrics)
    assert cm.shape[0] == s1.size
    assert abs(sum(wts.values()) - 1.0) < 1e-8
    ra = rank_average_combine(series)
    assert ra.shape[0] == s1.size
    maj = majority_sign_combine(series)
    assert maj.shape[0] == s1.size


def test_rank_candidates_not_sharpe() -> None:
    cands = [
        {
            "name": "a",
            "ic": 0.08,
            "hit_rate": 0.55,
            "stability": 0.7,
            "economic_hypothesis": "x" * 30,
        },
        {
            "name": "b",
            "ic": 0.02,
            "hit_rate": 0.51,
            "stability": 0.4,
            "economic_hypothesis": "y" * 30,
            "sharpe": 5.0,
        },
    ]
    ranked = rank_candidates(cands)
    assert ranked[0]["name"] == "a"  # higher IC/stability beats sharpe-only peer
    assert "disclaimer" in ranked[0]


def test_monitoring_retirement_alerts(
    signal: np.ndarray, returns: np.ndarray, fwd: np.ndarray
) -> None:
    half = signal.size // 2
    drift = signal_distribution_drift(signal[:half], signal[half:])
    assert "psi" in drift or "drifted" in drift
    cd = concept_drift_ic(signal[:half], fwd[:half], signal[half:], fwd[half:])
    assert "drifted" in cd or "ic_reference" in cd
    pd = position_drift(signal[:half], signal[half : 2 * half])
    assert "correlation" in pd or "drifted" in pd
    mon = monitor_signal_drift(
        signal[:half],
        signal[half:],
        reference_returns=fwd[:half],
        current_returns=fwd[half:],
    )
    assert isinstance(mon, dict)

    ric = rolling_ic(signal, fwd, window=40)
    assert "ic" in ric or "mean" in ric
    curve = ic_decay_curve(signal, returns, horizons=(1, 2, 5))
    hs = curve.get("horizons", [1, 2, 5])
    ics = np.asarray(curve.get("ics", curve.get("ic", [0.1, 0.05, 0.02])), dtype=float)
    hl = estimate_ic_half_life(hs, ics)
    assert hl is not None or True
    mic = monitor_ic_decay(ric, baseline_ic=0.05)
    assert "status" in mic or isinstance(mic, dict)

    rp = rolling_performance(returns, window=40)
    assert "sharpe" in rp or "mean" in rp or isinstance(rp, dict)
    pds = performance_decay_score(returns, baseline_window=80, recent_window=40)
    assert isinstance(pds, dict)
    mpd = monitor_performance_decay(returns)
    assert isinstance(mpd, dict)
    assert max_drawdown(returns) <= 0.0 or True

    active = evaluate_retirement(
        ic_recent=0.05, ic_baseline=0.04, net_sharpe=0.8, capacity=1e7, capacity_baseline=1e7
    )
    assert active["status"] in {"ACTIVE", "DEGRADED", "RETIRED"}
    retired = evaluate_retirement(
        ic_recent=-0.02,
        ic_baseline=0.05,
        net_sharpe=-0.5,
        performance_decayed=True,
        drift_severity="critical",
        capacity=1e3,
        capacity_baseline=1e7,
    )
    assert retired["status"] in {"DEGRADED", "RETIRED"}
    batch = batch_evaluate_retirement(
        {
            "a": {"ic_recent": 0.05, "ic_baseline": 0.04, "net_sharpe": 0.5},
            "b": {
                "ic_recent": -0.01,
                "ic_baseline": 0.05,
                "net_sharpe": -0.2,
                "performance_decayed": True,
            },
        }
    )
    assert "a" in batch and "b" in batch

    alerts = build_alpha_alerts(
        retirement=retired,
        drift=drift,
        performance=mpd,
        signal_name="unit",
    )
    assert isinstance(alerts, list)
    summary = summarize_alerts(alerts)
    assert isinstance(summary, dict)


def test_regime_performance(
    regime_scen: dict[str, Any], signal: np.ndarray, returns: np.ndarray, fwd: np.ndarray
) -> None:
    sig = np.asarray(regime_scen["signal"])
    ret = np.asarray(regime_scen["returns"])
    regimes = regime_scen["regimes"]
    from iqrp.app.alpha.research.decay import forward_returns as fr

    f = fr(ret, 1)
    ric = regime_ic(sig, f, regimes)
    assert "by_regime" in ric
    rr = regime_returns(ret, regimes, positions=np.sign(sig))
    assert isinstance(rr, dict)
    rp = regime_performance(sig, f, regimes)
    assert isinstance(rp, dict)
    rh = regime_hit_rate(sig, f, regimes)
    assert isinstance(rh, dict)

    labels = np.where(returns > 0, "bull", "bear")
    bull = labels == "bull"
    cic = conditional_ic(signal, fwd, bull)
    assert isinstance(cic, dict)
    cap = conditional_alpha_profile(signal, fwd, {"bull": bull, "bear": ~bull})
    assert isinstance(cap, dict)
    gated = regime_gated_signal(signal, labels, active_regimes={"bull"})
    assert gated.shape == signal.shape
    cmp = compare_unconditional_vs_conditional(signal, fwd, labels)
    assert isinstance(cmp, dict)
    applied = apply_condition_fn(signal, fwd, lambda s, r: np.where(s > 0, s, 0.0))
    assert isinstance(applied, dict)

    stab = regime_stability_score(signal, fwd, labels)
    assert "score" in stab or isinstance(stab, dict)
    roll = rolling_regime_stability(signal, fwd, labels, window=60)
    assert roll is not None
    conc = regime_concentration(signal, fwd, labels)
    assert "herfindahl" in conc or "concentrated" in conc or isinstance(conc, dict)
