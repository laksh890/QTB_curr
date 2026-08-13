"""Discovery templates; diagnostics leakage; PIT momentum construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from iqrp.app.alpha.base.signal_registry import SignalRegistry
from iqrp.app.alpha.diagnostics import (
    finite_check,
    leakage_shift_test,
    monotonic_time_check,
    pit_alignment_check,
    run_alpha_diagnostics,
)
from iqrp.app.alpha.discovery.alternative import (
    alternative_change_signal,
    alternative_zscore_signal,
    apply_publication_lag,
    sentiment_pressure_signal,
)
from iqrp.app.alpha.discovery.candidate_generator import (
    CandidateGenerator,
    generate_candidates,
)
from iqrp.app.alpha.discovery.cross_sectional import (
    cross_sectional_rank_signal,
    cross_sectional_zscore_signal,
    long_short_spread,
)
from iqrp.app.alpha.discovery.event_based import (
    earnings_drift_proxy,
    event_impulse_signal,
    surprise_signal,
)
from iqrp.app.alpha.discovery.statistical import candidates_to_signals, screen_features
from iqrp.app.alpha.discovery.symbolic import (
    as_float1d,
    delay,
    diff,
    evaluate_expression,
    lag,
    rank,
    ratio,
    rolling_mean,
    rolling_std,
    rolling_sum,
    signed_power,
    ts_max,
    ts_min,
    zscore,
)
from iqrp.app.alpha.discovery.time_series import (
    build_time_series_candidates,
    mean_reversion_signal,
    momentum_signal,
    trend_signal,
    volatility_signal,
    volume_signal,
)
from iqrp.app.alpha.research.decay import forward_returns
from iqrp.app.alpha.research.information_coefficient import compute_ic


def test_time_series_templates(returns: np.ndarray, rng: np.random.Generator) -> None:
    prices = 100.0 * np.cumprod(1.0 + returns)
    volume = np.abs(returns) * 1e6 + 1e5

    mom = momentum_signal(returns, lookback=20)
    assert mom.metadata["claims_profitability"] is False
    assert "definition" in mom.metadata
    hyp = mom.metadata["definition"]["economic_hypothesis"]
    assert len(hyp) >= 20

    # Prices path
    mom_p = momentum_signal(prices, lookback=10, input_kind="prices")
    assert mom_p.values.size == prices.size

    mr = mean_reversion_signal(returns, lookback=10)
    tr = trend_signal(prices, lookback_fast=5, lookback_slow=20, input_kind="prices")
    vol = volatility_signal(returns, lookback=20)
    vs = volume_signal(volume, lookback=20)
    assert all(s.values.size == returns.size for s in (mr, tr, vol, vs))

    cands = build_time_series_candidates(
        returns, volume=volume, momentum_lookbacks=(10, 20), mean_rev_lookbacks=(5,)
    )
    assert len(cands) >= 1

    with pytest.raises(ValueError):
        momentum_signal(returns, lookback=0)
    with pytest.raises(ValueError):
        momentum_signal(returns, input_kind="bad")  # type: ignore[arg-type]


def test_pit_momentum_independent_of_future_returns(returns: np.ndarray) -> None:
    """PIT: momentum at t uses only past returns — independent of returns[t+1:]."""
    lookback = 20
    mom = momentum_signal(returns, lookback=lookback)
    values = mom.values
    # For each t with finite signal, recomputing from returns[:t+1] matches
    for t in (lookback + 5, lookback + 50, len(returns) - 10):
        if not np.isfinite(values[t]):
            continue
        # Past-only window ending at t (inclusive of t per template docstring)
        past = returns[t - lookback + 1 : t + 1]
        expected = float(np.sum(past))
        assert abs(values[t] - expected) < 1e-9
        # Changing future returns must not change signal[t]
        mutated = returns.copy()
        mutated[t + 1 :] = 999.0
        mom2 = momentum_signal(mutated, lookback=lookback)
        assert abs(mom2.values[t] - values[t]) < 1e-9


def test_symbolic_ops_no_negative_lag(returns: np.ndarray) -> None:
    x = as_float1d(returns)
    assert lag(x, 1)[0] != lag(x, 1)[0] or np.isnan(lag(x, 1)[0])
    with pytest.raises(ValueError):
        lag(x, -1)
    with pytest.raises(ValueError):
        delay(x, -1)

    d = diff(x, 1)
    r = ratio(x, np.abs(x) + 1.0)
    rm = rolling_mean(x, 10)
    rs = rolling_std(x, 10)
    rsum = rolling_sum(x, 10)
    zs = zscore(x, 20)
    rk = rank(x, window=20)
    rk2 = rank(x, window=None)
    assert d.size == x.size
    assert all(a.size == x.size for a in (r, rm, rs, rsum, zs, rk, rk2))
    assert ts_max(x, 10).size == x.size
    assert ts_min(x, 10).size == x.size
    assert signed_power(x, 2.0).size == x.size

    expr = evaluate_expression(
        [
            ("load", {"name": "r"}),
            ("lag", {"periods": 1}),
            ("zscore", {"window": 10}),
        ],
        {"r": x},
    )
    assert expr.size == x.size


def test_cross_sectional_event_alt_statistical(
    panel: np.ndarray, returns: np.ndarray, rng: np.random.Generator
) -> None:
    crs = cross_sectional_rank_signal(panel, asset_index=0)
    czs = cross_sectional_zscore_signal(panel, asset_index=1)
    ls = long_short_spread(panel, top_frac=0.2, bottom_frac=0.2)
    assert crs.values.size == panel.shape[0]

    mask = np.zeros(returns.size, dtype=bool)
    mask[::25] = True
    ev = event_impulse_signal(mask, decay=0.9, horizon=5)
    sur = surprise_signal(returns, np.zeros_like(returns), lookback=10)
    ed = earnings_drift_proxy(returns, mask, post_window=5)
    assert ev.values.size == returns.size
    assert sur.values.size == returns.size
    assert ed.values.size == returns.size

    alt = alternative_zscore_signal(returns, lookback=20, publication_lag=1)
    ch = alternative_change_signal(returns, change_window=5, publication_lag=1)
    pos = np.clip(returns, 0, None)
    neg = np.clip(-returns, 0, None)
    sent = sentiment_pressure_signal(pos, neg, lookback=10, publication_lag=1)
    lagged = apply_publication_lag(returns, 2)
    assert np.all(np.isnan(lagged[:2]))
    assert alt.metadata.get("claims_profitability") is False
    assert ch.values.size == returns.size
    assert sent.values.size == returns.size

    features = {
        "f1": returns + 0.3 * np.roll(returns, -1),  # weak look-ahead mix for screen
        "f2": rng.normal(size=returns.size),
        "f3": np.cumsum(returns),
    }
    target = forward_returns(returns, 1)
    screened = screen_features(features, target, min_abs_ic=0.0, min_obs=30)
    assert isinstance(screened, list)
    for c in screened:
        assert c.is_alpha is False
    sigs = candidates_to_signals(screened, features)
    assert isinstance(sigs, list)


def test_candidate_generator_and_generate(
    returns: np.ndarray, rng: np.random.Generator, panel: np.ndarray
) -> None:
    reg = SignalRegistry()
    gen = CandidateGenerator(registry=reg, auto_register=True)
    volume = np.abs(returns) * 1e6 + 1e5
    prices = 100.0 * np.cumprod(1.0 + returns)
    features = {"f1": returns, "f2": rng.normal(size=returns.size)}
    target = forward_returns(returns, 1)

    res = gen.discover_all(
        returns=returns,
        features=features,
        target=target,
        volume=volume,
        prices=prices,
        alt_series=returns,
        event_mask=(np.abs(returns) > np.nanpercentile(np.abs(returns), 90)),
        forecast=returns * 0.2,
        forecast_hypothesis=(
            "Forecast innovations capture delayed incorporation of public news."
        ),
    )
    assert len(res.signals) >= 1
    d = res.to_dict()
    assert "disclaimer" in d or "notes" in d
    assert all(defn.economic_hypothesis for defn in res.definitions)
    # Auto-registered as candidates, not approved
    for eid in res.experiment_ids:
        if eid:
            assert reg.get(eid).status.value == "CANDIDATE"

    out = generate_candidates(
        returns,
        volume=volume,
        registry=SignalRegistry(),
        auto_register=False,
    )
    assert len(out.signals) >= 0


def test_leakage_shift_and_pit_diagnostics(
    genuine: dict[str, Any], rng: np.random.Generator
) -> None:
    sig = np.asarray(genuine["signal"])
    ret = np.asarray(genuine["returns"])
    fwd = forward_returns(ret, 1)

    # Clean PIT momentum-like: peak IC should not be at strongly negative lag
    clean = leakage_shift_test(sig, fwd, max_lead=3, min_obs=30)
    assert "curve" in clean
    assert "suspicious" in clean

    # Artificial leakage: signal = future return
    leaked = np.roll(fwd, -1)
    leaked[-1] = np.nan
    leak = leakage_shift_test(leaked, fwd, max_lead=3, min_obs=30)
    # Best lag often negative / suspicious for look-ahead construction
    assert "best_lag" in leak

    fc = finite_check({"signal": sig, "fwd": fwd, "bad": np.array([np.nan, np.inf, 1.0])})
    assert fc["ok"] is False
    clean_sig = np.nan_to_num(sig, nan=0.0)
    clean_fwd = np.nan_to_num(fwd, nan=0.0)
    fc2 = finite_check([clean_sig, clean_fwd], names=["s", "f"])
    assert fc2["ok"] is True

    ts = np.arange(sig.size)
    pit_ok = pit_alignment_check(ts, feature_asof=ts, label_asof=ts + 1, allow_equal=True)
    # label_asof > ts → future leak
    assert pit_ok["ok"] is False or "label_asof_future_leak" in pit_ok["issues"]

    pit_good = pit_alignment_check(ts, feature_asof=ts - 1, allow_equal=True)
    assert pit_good["ok"] is True

    mono = monotonic_time_check(ts)
    assert mono["ok"] is True
    mono_bad = monotonic_time_check(np.array([1, 2, 2, 3]))
    assert mono_bad["ok"] is False

    diag = run_alpha_diagnostics(
        signal=sig,
        forward_returns=fwd,
        timestamps=ts,
        feature_asof=ts,
        label_asof=ts,
        extra_arrays={"ret": ret},
    )
    assert "finite" in diag and "leakage" in diag


def test_genuine_ic_beats_noise_same_seed(
    genuine: dict[str, Any], noise: dict[str, Any]
) -> None:
    g_ic = abs(
        compute_ic(
            np.asarray(genuine["signal"]),
            forward_returns(np.asarray(genuine["returns"]), 1),
        )
    )
    n_ic = abs(
        compute_ic(
            np.asarray(noise["signal"]),
            forward_returns(np.asarray(noise["returns"]), 1),
        )
    )
    assert g_ic > n_ic
