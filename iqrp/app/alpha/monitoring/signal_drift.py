"""Signal / feature / concept drift monitors."""

from __future__ import annotations

from typing import Any

import numpy as np


def _psi(expected: np.ndarray, actual: np.ndarray, *, bins: int = 10) -> float:
    e = expected[np.isfinite(expected)]
    a = actual[np.isfinite(actual)]
    if e.size < 5 or a.size < 5:
        return float("nan")
    qs = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(e, qs))
    if edges.size < 3:
        return float("nan")
    e_hist, _ = np.histogram(e, bins=edges)
    a_hist, _ = np.histogram(a, bins=edges)
    e_pct = e_hist / max(e_hist.sum(), 1)
    a_pct = a_hist / max(a_hist.sum(), 1)
    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(a[np.isfinite(a)])
    y = np.sort(b[np.isfinite(b)])
    if x.size < 5 or y.size < 5:
        return float("nan")
    # two-sample KS via empirical CDF on pooled grid
    grid = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, grid, side="right") / x.size
    cdf_y = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def signal_distribution_drift(
    reference: Any,
    current: Any,
    *,
    psi_threshold: float = 0.25,
    ks_threshold: float = 0.20,
) -> dict[str, Any]:
    """Population / covariate drift between reference and current windows."""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cur = np.asarray(current, dtype=np.float64).reshape(-1)
    psi = _psi(ref, cur)
    ks = _ks(ref, cur)
    ref_f = ref[np.isfinite(ref)]
    cur_f = cur[np.isfinite(cur)]
    mean_z = float("nan")
    if ref_f.size and cur_f.size:
        sd = float(np.std(ref_f))
        if sd > 1e-12:
            mean_z = abs(float(np.mean(cur_f) - np.mean(ref_f))) / sd

    alerts: list[str] = []
    if np.isfinite(psi) and psi >= psi_threshold:
        alerts.append("psi_breach")
    if np.isfinite(ks) and ks >= ks_threshold:
        alerts.append("ks_breach")
    if np.isfinite(mean_z) and mean_z >= 2.0:
        alerts.append("mean_shift")

    severity = "none"
    if alerts:
        severity = "high" if len(alerts) >= 2 or (np.isfinite(psi) and psi >= 2 * psi_threshold) else "medium"

    return {
        "name": "signal_distribution_drift",
        "psi": psi,
        "ks": ks,
        "mean_z": mean_z,
        "alerts": alerts,
        "severity": severity,
        "drifted": len(alerts) > 0,
    }


def concept_drift_ic(
    signal_ref: Any,
    returns_ref: Any,
    signal_cur: Any,
    returns_cur: Any,
    *,
    ratio_threshold: float = 0.5,
) -> dict[str, Any]:
    """Concept drift via IC ratio current/reference."""
    def _ic(s: Any, r: Any) -> float:
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        m = np.isfinite(s) & np.isfinite(r)
        if m.sum() < 3:
            return float("nan")
        a, b = s[m], r[m]
        if np.std(a) < 1e-15 or np.std(b) < 1e-15:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    ic_ref = _ic(signal_ref, returns_ref)
    ic_cur = _ic(signal_cur, returns_cur)
    if np.isfinite(ic_ref) and abs(ic_ref) > 1e-9:
        ratio = abs(ic_cur) / abs(ic_ref)
        sign_flip = np.sign(ic_cur) != np.sign(ic_ref) and abs(ic_cur) > 1e-6
    else:
        ratio = float("nan")
        sign_flip = False

    drifted = (np.isfinite(ratio) and ratio < ratio_threshold) or sign_flip
    return {
        "name": "concept_drift_ic",
        "ic_reference": ic_ref,
        "ic_current": ic_cur,
        "ratio": float(ratio) if np.isfinite(ratio) else float("nan"),
        "sign_flip": bool(sign_flip),
        "drifted": bool(drifted),
        "severity": "high" if sign_flip or (np.isfinite(ratio) and ratio < 0.3) else ("medium" if drifted else "none"),
    }


def position_drift(
    positions_ref: Any,
    positions_cur: Any,
    *,
    corr_threshold: float = 0.7,
) -> dict[str, Any]:
    """Detect drift in position / exposure profiles."""
    a = np.asarray(positions_ref, dtype=np.float64).reshape(-1)
    b = np.asarray(positions_cur, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    corr = float("nan")
    if m.sum() >= 3 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
        corr = float(np.corrcoef(a[m], b[m])[0, 1])
    l1 = float(np.nanmean(np.abs(a - b))) if n else float("nan")
    drifted = (np.isfinite(corr) and corr < corr_threshold) or (np.isfinite(l1) and l1 > 0.5)
    return {
        "name": "position_drift",
        "correlation": corr,
        "mean_abs_delta": l1,
        "drifted": bool(drifted),
        "severity": "high" if drifted and (not np.isfinite(corr) or corr < 0.4) else ("medium" if drifted else "none"),
    }


def monitor_signal_drift(
    reference_signal: Any,
    current_signal: Any,
    *,
    reference_returns: Any | None = None,
    current_returns: Any | None = None,
) -> dict[str, Any]:
    """Aggregate drift monitors."""
    dist = signal_distribution_drift(reference_signal, current_signal)
    out: dict[str, Any] = {
        "name": "monitor_signal_drift",
        "distribution": dist,
        "drifted": bool(dist["drifted"]),
        "severity": dist["severity"],
    }
    if reference_returns is not None and current_returns is not None:
        concept = concept_drift_ic(
            reference_signal, reference_returns, current_signal, current_returns
        )
        out["concept"] = concept
        out["drifted"] = out["drifted"] or concept["drifted"]
        if concept["severity"] == "high" or out["severity"] == "none":
            out["severity"] = concept["severity"]
    return out
