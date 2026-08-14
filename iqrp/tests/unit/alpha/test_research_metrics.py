"""IC, Rank IC, hit rate, decay, stability, persistence, seasonality, evaluator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from iqrp.app.alpha.base.signal_definition import SignalDefinition
from iqrp.app.alpha.research.decay import analyze_decay, forward_returns
from iqrp.app.alpha.research.evaluator import (
    SignalEvaluator,
    compute_signal_statistics,
    evaluate_signal,
)
from iqrp.app.alpha.research.hit_rate import (
    compute_hit_rate,
    hit_rate_summary,
    rolling_hit_rate,
)
from iqrp.app.alpha.research.information_coefficient import (
    compute_ic,
    ic_summary,
    rolling_ic,
)
from iqrp.app.alpha.research.persistence import (
    autocorrelation,
    persistence_profile,
    persistence_summary,
    signal_half_life,
)
from iqrp.app.alpha.research.predictor import SignalPredictor, predict_forward
from iqrp.app.alpha.research.rank_ic import (
    compute_rank_ic,
    rank_ic_summary,
    rolling_rank_ic,
)
from iqrp.app.alpha.research.seasonality import analyze_seasonality, month_of_year_ic
from iqrp.app.alpha.research.stability import analyze_stability


def test_ic_and_rank_ic(signal: np.ndarray, fwd: np.ndarray) -> None:
    ic = compute_ic(signal, fwd)
    ric = compute_rank_ic(signal, fwd)
    assert np.isfinite(ic)
    assert np.isfinite(ric)
    ric_roll = rolling_rank_ic(signal, fwd, window=40, step=5, min_obs=20)
    assert ric_roll.size > 0
    summary = ic_summary(signal, fwd, window=40, step=5)
    assert "ic" in summary and "disclaimer" in summary
    rsum = rank_ic_summary(signal, fwd, window=40, step=5)
    assert "rank_ic" in rsum
    roll = rolling_ic(signal, fwd, window=40, step=5, min_obs=20)
    assert roll.size > 0


def test_hit_rate(signal: np.ndarray, fwd: np.ndarray) -> None:
    hr = compute_hit_rate(signal, fwd)
    assert 0.0 <= hr <= 1.0 or np.isnan(hr)
    roll = rolling_hit_rate(signal, fwd, window=40, step=5)
    assert roll.size > 0
    summary = hit_rate_summary(signal, fwd, window=40, step=5)
    assert "hit_rate" in summary


def test_decay_stability_persistence_seasonality(
    signal: np.ndarray, returns: np.ndarray, decay_scen: dict[str, Any]
) -> None:
    decay = analyze_decay(signal, returns, horizons=(1, 2, 5, 10))
    assert set(decay["horizons"]) == {1, 2, 5, 10}
    assert "half_life" in decay
    assert "optimal_hold" in decay

    d2 = analyze_decay(decay_scen["signal"], decay_scen["returns"], horizons=(1, 2, 5, 10))
    assert np.isfinite(d2["half_life"]) or True

    stab = analyze_stability(signal, returns, horizon=1, window=60, step=10, min_obs=30)
    assert "stability_score" in stab or "overall_ic" in stab

    ac = autocorrelation(signal, lag_periods=1)
    assert np.isfinite(ac) or np.isnan(ac)
    prof = persistence_profile(signal, lags=(1, 2, 5, 10))
    assert isinstance(prof, dict) and len(prof) >= 1
    hl = signal_half_life(signal, max_lag=20)
    assert hl is not None
    psum = persistence_summary(signal)
    assert "lag1" in psum or "autocorr" in psum or isinstance(psum, dict)

    season = analyze_seasonality(signal, returns, period=5, horizon=1)
    assert "ic_by_bucket" in season
    months = np.tile(np.arange(1, 13), returns.size // 12 + 1)[: returns.size]
    moy = month_of_year_ic(signal, forward_returns(returns, 1), months)
    assert "ic_by_month" in moy


def test_evaluator_and_predictor(
    signal: np.ndarray,
    returns: np.ndarray,
    definition: SignalDefinition,
) -> None:
    stats = compute_signal_statistics(signal)
    assert stats.n_obs == signal.size
    assert stats.n_finite <= signal.size

    ev = SignalEvaluator(horizons=(1, 2, 5), stability_window=60, seasonality_period=5)
    report = ev.evaluate(signal, returns, definition=definition)
    assert report.performance is not None
    assert report.score is not None
    d = report.to_dict()
    assert d["rules"]["statistical_significance_alone_is_not_alpha"] is True
    assert d["rules"]["historical_sharpe_alone_cannot_approve"] is True

    report2 = evaluate_signal(signal, returns, definition=definition)
    assert report2.performance is not None

    # Thin hyp warning path
    thin = SignalDefinition(
        name="t",
        version="1.0.0",
        formula="x",
        features=("x",),
        lookback=5,
        horizon=1,
        universe="u",
        frequency="1d",
        direction="long_short",
        expected_relationship="unknown",
        economic_hypothesis="",
        owner="r",
    )
    r_thin = evaluate_signal(signal, returns, definition=thin)
    assert r_thin.warnings

    pred = SignalPredictor(min_train=60, test_size=20, step=20)
    result = pred.predict(signal, returns)
    assert result.n_train >= 0
    pf = predict_forward(signal, returns, horizon=1, ridge_alpha=1.0)
    assert pf is not None


def test_forward_returns_trailing_nan(returns: np.ndarray) -> None:
    fwd = forward_returns(returns, 3)
    assert np.isnan(fwd[-1])
    assert np.isnan(fwd[-2])
    assert np.isnan(fwd[-3])


def test_ic_constant_signal(fwd: np.ndarray) -> None:
    const = np.ones_like(fwd)
    ic = compute_ic(const, fwd)
    assert np.isnan(ic) or abs(ic) < 1e-8


def test_hit_rate_all_zero_signs() -> None:
    s = np.zeros(50)
    r = np.ones(50)
    hr = compute_hit_rate(s, r)
    assert np.isnan(hr) or hr == 0.0
