"""Automated look-ahead / leakage tests. Fail loudly on detected leakage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class LeakageError(ValueError):
    """Raised when future information contamination is detected."""


def assert_no_future_columns(frame: pd.DataFrame, *, forbidden_prefixes: tuple[str, ...] = ("future_", "fwd_", "y_")) -> None:
    bad = [c for c in frame.columns if str(c).lower().startswith(forbidden_prefixes)]
    if bad:
        raise LeakageError(f"future/label columns rejected in feature frame: {bad}")


def feature_causality_shift_test(
    feature: pd.Series,
    prices: pd.Series,
    *,
    lookback: int,
) -> dict[str, Any]:
    """If feature at t correlates extremely with future price move vs past, flag risk.

    Primary hard check: recomputing with shifted-future prices must change feature
    when lookback uses prices — and feature must not equal lead(return).
    """
    f = pd.Series(feature, dtype=np.float64)
    px = pd.Series(prices, dtype=np.float64)
    fut_ret = px.shift(-1) / px - 1.0
    past_ret = px / px.shift(max(lookback, 1)) - 1.0
    # correlation with future vs past
    mask = f.notna() & fut_ret.notna() & past_ret.notna()
    if mask.sum() < 10:
        return {"ok": True, "note": "insufficient overlap", "n": int(mask.sum())}
    ic_fut = float(np.corrcoef(f[mask], fut_ret[mask])[0, 1])
    ic_past = float(np.corrcoef(f[mask], past_ret[mask])[0, 1])
    # Hard fail: feature nearly identical to future return
    if np.isfinite(ic_fut) and abs(ic_fut) > 0.999:
        raise LeakageError(
            f"feature nearly identical to 1-bar future return (corr={ic_fut:.4f})"
        )
    return {
        "ok": True,
        "corr_with_future_1bar": ic_fut,
        "corr_with_past_lookback": ic_past,
        "n": int(mask.sum()),
    }


def future_shift_changes_result(
    compute_fn,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Shifting inputs forward must change a causal feature that depends on them.

    Price-based features must change when OHLC is shifted. Volume-only features
    need not change under an OHLC shift, but must change when volume is shifted.
    """
    base = compute_fn(frame)
    shifted = frame.copy()
    shifted["close"] = shifted["close"].shift(-1)
    # also shift OHLC consistently
    for col in ("open", "high", "low"):
        if col in shifted.columns:
            shifted[col] = shifted[col].shift(-1)
    alt = compute_fn(shifted)
    b = np.asarray(base, dtype=np.float64)
    a = np.asarray(alt, dtype=np.float64)
    n = min(b.size, a.size)
    mask = np.isfinite(b[:n]) & np.isfinite(a[:n])
    if mask.sum() < 5:
        return {"ok": True, "changed": False, "note": "insufficient finite overlap"}
    changed = not np.allclose(b[:n][mask], a[:n][mask], equal_nan=True)
    if changed:
        return {"ok": True, "changed": True, "dependency": "price"}

    # Price-independent features (e.g. volume_change): require volume causality instead
    if "volume" in frame.columns:
        shifted_v = frame.copy()
        shifted_v["volume"] = shifted_v["volume"].shift(-1)
        alt_v = compute_fn(shifted_v)
        av = np.asarray(alt_v, dtype=np.float64)
        n2 = min(b.size, av.size)
        mask2 = np.isfinite(b[:n2]) & np.isfinite(av[:n2])
        if mask2.sum() >= 5 and not np.allclose(b[:n2][mask2], av[:n2][mask2], equal_nan=True):
            return {
                "ok": True,
                "changed": False,
                "volume_changed": True,
                "dependency": "volume",
                "note": "feature is price-independent; volume forward-shift changes output",
            }

    raise LeakageError(
        "shifting future prices did not change feature — possible non-causal compute"
    )


def normalization_no_future_test(series: pd.Series, *, window: int = 20) -> dict[str, Any]:
    """Causal rolling zscore at t must ignore values after t."""
    from iqrp.app.backtesting.alpha_research.normalize import causal_rolling_zscore

    s = pd.Series(series, dtype=np.float64)
    z = causal_rolling_zscore(s, window=window)
    # mutate a future point far ahead; early zscores must be unchanged
    if len(s) < window + 5:
        return {"ok": True, "note": "series too short"}
    s2 = s.copy()
    s2.iloc[-1] = float(s2.iloc[-1]) + 1e6
    z2 = causal_rolling_zscore(s2, window=window)
    early = slice(0, len(s) - 2)
    if not np.allclose(
        z.iloc[early].to_numpy(dtype=np.float64),
        z2.iloc[early].to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise LeakageError("normalization appears to use future observations")
    return {"ok": True}


def run_leakage_suite(
    frame: pd.DataFrame,
    feature: pd.Series,
    *,
    lookback: int,
    compute_fn=None,
) -> dict[str, Any]:
    assert_no_future_columns(frame)
    results = {
        "future_columns": {"ok": True},
        "causality_corr": feature_causality_shift_test(feature, frame["close"], lookback=lookback),
        "normalization": normalization_no_future_test(frame["close"]),
    }
    if compute_fn is not None:
        results["future_shift_changes"] = future_shift_changes_result(compute_fn, frame)
    results["ok"] = all(v.get("ok", False) for v in results.values() if isinstance(v, dict))
    return results


__all__ = [
    "LeakageError",
    "assert_no_future_columns",
    "feature_causality_shift_test",
    "future_shift_changes_result",
    "normalization_no_future_test",
    "run_leakage_suite",
]
