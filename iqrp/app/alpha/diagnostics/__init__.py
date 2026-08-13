"""Alpha diagnostics: leakage, finiteness, and point-in-time (PIT) checks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def finite_check(
    arrays: Mapping[str, Any] | Sequence[Any],
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Report non-finite counts / fractions for named arrays."""
    if isinstance(arrays, Mapping):
        items = list(arrays.items())
    else:
        labels = list(names) if names is not None else [f"arr_{i}" for i in range(len(arrays))]
        items = list(zip(labels, arrays))

    details: dict[str, Any] = {}
    issues: list[str] = []
    for name, val in items:
        arr = np.asarray(val, dtype=np.float64)
        n = int(arr.size)
        n_non = int(np.sum(~np.isfinite(arr))) if n else 0
        details[str(name)] = {
            "shape": list(arr.shape),
            "n": n,
            "n_nonfinite": n_non,
            "frac_nonfinite": float(n_non / n) if n else 0.0,
            "n_nan": int(np.sum(np.isnan(arr))) if n else 0,
            "n_inf": int(np.sum(np.isinf(arr))) if n else 0,
        }
        if n_non > 0:
            issues.append(f"{name}_nonfinite")

    return {
        "name": "finite_check",
        "ok": len(issues) == 0,
        "issues": issues,
        "details": details,
    }


def leakage_shift_test(
    signal: Any,
    forward_returns: Any,
    *,
    max_lead: int = 5,
    min_obs: int = 30,
) -> dict[str, Any]:
    """Shift test for look-ahead leakage.

    Computes IC(signal_t, return_{t+k}) for k in ``[-max_lead, +max_lead]``.
    Suspicious if peak |IC| occurs at negative k (signal aligned with future returns
    that should not yet be known when using contemporaneous indexing conventions),
    or if IC at k=0 dominates unrealistically vs positive lags when signal is claimed
    predictive for future returns — we flag when best lag is negative.
    """
    sig = np.asarray(signal, dtype=np.float64)
    ret = np.asarray(forward_returns, dtype=np.float64)

    def _series_ic(s: np.ndarray, r: np.ndarray) -> float:
        if s.ndim == 2:
            daily = []
            for i in range(min(s.shape[0], r.shape[0])):
                a, b = s[i], r[i] if r.ndim == 2 else r
                if r.ndim == 1:
                    continue
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 3:
                    continue
                aa, bb = a[m], b[m]
                if np.std(aa) < 1e-15 or np.std(bb) < 1e-15:
                    continue
                daily.append(float(np.corrcoef(aa, bb)[0, 1]))
            return float(np.nanmean(daily)) if daily else float("nan")
        s1 = s.reshape(-1)
        r1 = r.reshape(-1) if r.ndim == 1 else np.nanmean(r, axis=1)
        n = min(s1.size, r1.size)
        s1, r1 = s1[:n], r1[:n]
        m = np.isfinite(s1) & np.isfinite(r1)
        if int(m.sum()) < min_obs:
            return float("nan")
        a, b = s1[m], r1[m]
        if np.std(a) < 1e-15 or np.std(b) < 1e-15:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    lags = list(range(-int(max_lead), int(max_lead) + 1))
    curve: list[dict[str, Any]] = []
    for k in lags:
        if k == 0:
            ic = _series_ic(sig, ret)
        elif k > 0:
            # signal leads returns by k (expected for predictive alpha)
            if sig.ndim == 1:
                ic = _series_ic(sig[:-k], ret[k:] if ret.ndim == 1 else ret[k:])
            else:
                ic = _series_ic(sig[:-k], ret[k:])
        else:
            # negative: returns lead signal → potential leakage if strongest
            kk = -k
            if sig.ndim == 1:
                ic = _series_ic(sig[kk:], ret[:-kk] if ret.ndim == 1 else ret[:-kk])
            else:
                ic = _series_ic(sig[kk:], ret[:-kk])
        curve.append({"lag": k, "ic": float(ic) if np.isfinite(ic) else float("nan")})

    finite = [(c["lag"], c["ic"]) for c in curve if np.isfinite(c["ic"])]
    if finite:
        best_lag, best_ic = max(finite, key=lambda t: abs(t[1]))
    else:
        best_lag, best_ic = 0, float("nan")

    suspicious = bool(best_lag < 0 and np.isfinite(best_ic) and abs(best_ic) > 0.02)
    return {
        "name": "leakage_shift_test",
        "curve": curve,
        "best_lag": int(best_lag),
        "best_ic": float(best_ic) if np.isfinite(best_ic) else float("nan"),
        "suspicious": suspicious,
        "ok": not suspicious,
        "message": (
            f"Peak |IC| at lag={best_lag} suggests possible look-ahead"
            if suspicious
            else "No leakage signature detected by shift test"
        ),
    }


def pit_alignment_check(
    timestamps: Any,
    *,
    feature_asof: Any | None = None,
    label_asof: Any | None = None,
    universe_asof: Any | None = None,
    allow_equal: bool = True,
) -> dict[str, Any]:
    """Point-in-time alignment checks: as-of timestamps must not exceed event time."""
    ts = np.asarray(timestamps)
    issues: list[str] = []
    details: dict[str, Any] = {"n": int(ts.size)}

    def _check(name: str, asof: Any) -> None:
        a = np.asarray(asof)
        if a.shape[0] != ts.shape[0]:
            issues.append(f"{name}_length_mismatch")
            details[name] = {"ok": False, "reason": "length_mismatch"}
            return
        # numeric or datetime64 comparable
        try:
            if allow_equal:
                bad = a > ts
            else:
                bad = a >= ts
            n_bad = int(np.sum(bad))
        except TypeError:
            issues.append(f"{name}_incomparable")
            details[name] = {"ok": False, "reason": "incomparable_dtypes"}
            return
        details[name] = {"ok": n_bad == 0, "n_future": n_bad}
        if n_bad > 0:
            issues.append(f"{name}_future_leak")

    if feature_asof is not None:
        _check("feature_asof", feature_asof)
    if label_asof is not None:
        _check("label_asof", label_asof)
    if universe_asof is not None:
        _check("universe_asof", universe_asof)

    return {
        "name": "pit_alignment_check",
        "ok": len(issues) == 0,
        "issues": issues,
        "details": details,
    }


def monotonic_time_check(timestamps: Any) -> dict[str, Any]:
    """Ensure timestamps are strictly increasing (no shuffle / duplication)."""
    ts = np.asarray(timestamps)
    if ts.size <= 1:
        return {"name": "monotonic_time_check", "ok": True, "n_violations": 0}
    diffs_ok = ts[1:] > ts[:-1]
    n_bad = int(np.sum(~diffs_ok))
    return {
        "name": "monotonic_time_check",
        "ok": n_bad == 0,
        "n_violations": n_bad,
        "issues": ["non_monotonic_timestamps"] if n_bad else [],
    }


def run_alpha_diagnostics(
    *,
    signal: Any | None = None,
    forward_returns: Any | None = None,
    timestamps: Any | None = None,
    feature_asof: Any | None = None,
    label_asof: Any | None = None,
    universe_asof: Any | None = None,
    extra_arrays: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate finite, leakage, and PIT diagnostics."""
    arrays: dict[str, Any] = {}
    if signal is not None:
        arrays["signal"] = signal
    if forward_returns is not None:
        arrays["forward_returns"] = forward_returns
    if extra_arrays:
        arrays.update(extra_arrays)

    finite = finite_check(arrays) if arrays else {"name": "finite_check", "ok": True, "issues": [], "details": {}}
    leakage = (
        leakage_shift_test(signal, forward_returns)
        if signal is not None and forward_returns is not None
        else None
    )
    pit = (
        pit_alignment_check(
            timestamps,
            feature_asof=feature_asof,
            label_asof=label_asof,
            universe_asof=universe_asof,
        )
        if timestamps is not None
        else None
    )
    mono = monotonic_time_check(timestamps) if timestamps is not None else None

    ok = bool(finite.get("ok", True))
    if leakage is not None:
        ok = ok and bool(leakage.get("ok", True))
    if pit is not None:
        ok = ok and bool(pit.get("ok", True))
    if mono is not None:
        ok = ok and bool(mono.get("ok", True))

    return {
        "name": "alpha_diagnostics",
        "ok": ok,
        "finite": finite,
        "leakage": leakage,
        "pit": pit,
        "monotonic_time": mono,
    }
