"""IC and return performance conditioned on regime labels."""

from __future__ import annotations

from typing import Any

import numpy as np


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan")
    a, b = x[m], y[m]
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan")
    a, b = x[m], y[m]
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    if np.std(ra) < 1e-15 or np.std(rb) < 1e-15:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _as_1d(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _align_regime_labels(regimes: Any, n: int) -> np.ndarray:
    labels = np.asarray(regimes)
    if labels.shape[0] != n:
        raise ValueError(f"regimes length {labels.shape[0]} != series length {n}")
    return labels


def regime_ic(
    signal: Any,
    forward_returns: Any,
    regimes: Any,
    *,
    rank: bool = False,
) -> dict[str, Any]:
    """Compute mean IC within each regime label.

    For 1D series, IC is time-series correlation within regime slices.
    For 2D panels ``(T, N)``, IC is the mean cross-sectional IC on dates in each regime.
    """
    sig = np.asarray(signal, dtype=np.float64)
    ret = np.asarray(forward_returns, dtype=np.float64)
    if sig.shape != ret.shape:
        raise ValueError("signal and forward_returns shape mismatch")

    corr_fn = _spearman if rank else _pearson
    by_regime: dict[str, dict[str, Any]] = {}

    if sig.ndim == 1:
        labels = _align_regime_labels(regimes, sig.shape[0])
        for g in sorted({str(x) for x in labels.tolist()}, key=str):
            mask = np.asarray([str(x) == g for x in labels.tolist()])
            ic = corr_fn(sig[mask], ret[mask])
            by_regime[g] = {
                "ic": ic,
                "n_obs": int(mask.sum()),
                "mean_return": float(np.nanmean(ret[mask])) if mask.any() else float("nan"),
            }
    else:
        if sig.ndim != 2:
            raise ValueError("signal must be 1D or 2D")
        labels = _align_regime_labels(regimes, sig.shape[0])
        for g in sorted({str(x) for x in labels.tolist()}, key=str):
            idx = np.asarray([str(x) == g for x in labels.tolist()])
            ics: list[float] = []
            rets: list[float] = []
            for i in np.where(idx)[0]:
                ics.append(corr_fn(sig[i], ret[i]))
                rets.append(float(np.nanmean(ret[i])))
            arr = np.asarray(ics, dtype=np.float64)
            by_regime[g] = {
                "ic": float(np.nanmean(arr)) if arr.size else float("nan"),
                "ic_std": float(np.nanstd(arr)) if arr.size else float("nan"),
                "n_dates": int(idx.sum()),
                "mean_return": float(np.nanmean(rets)) if rets else float("nan"),
            }

    ics_all = [v["ic"] for v in by_regime.values() if np.isfinite(v["ic"])]
    return {
        "name": "regime_ic",
        "rank": rank,
        "by_regime": by_regime,
        "n_regimes": len(by_regime),
        "ic_dispersion": float(np.std(ics_all)) if len(ics_all) > 1 else 0.0,
        "mean_abs_ic": float(np.mean(np.abs(ics_all))) if ics_all else float("nan"),
    }


def regime_returns(
    returns: Any,
    regimes: Any,
    *,
    positions: Any | None = None,
) -> dict[str, Any]:
    """Aggregate returns (or position-weighted returns) by regime."""
    r = _as_1d(returns)
    labels = _align_regime_labels(regimes, r.shape[0])
    if positions is not None:
        pos = _as_1d(positions)
        if pos.shape[0] != r.shape[0]:
            raise ValueError("positions length mismatch")
        pnl = pos * r
    else:
        pnl = r

    by_regime: dict[str, dict[str, Any]] = {}
    for g in sorted({str(x) for x in labels.tolist()}, key=str):
        mask = np.asarray([str(x) == g for x in labels.tolist()])
        x = pnl[mask]
        x = x[np.isfinite(x)]
        mu = float(np.mean(x)) if x.size else float("nan")
        sd = float(np.std(x, ddof=1)) if x.size > 1 else float("nan")
        sharpe = mu / sd * np.sqrt(252.0) if np.isfinite(sd) and sd > 1e-15 else float("nan")
        by_regime[g] = {
            "mean_return": mu,
            "vol": sd,
            "sharpe": float(sharpe) if np.isfinite(sharpe) else float("nan"),
            "n_obs": int(x.size),
            "total_return": float(np.sum(x)) if x.size else float("nan"),
        }
    return {"name": "regime_returns", "by_regime": by_regime}


def regime_performance(
    signal: Any,
    forward_returns: Any,
    regimes: Any,
    *,
    strategy_returns: Any | None = None,
) -> dict[str, Any]:
    """Combined IC + return scorecard by regime."""
    ic_rep = regime_ic(signal, forward_returns, regimes, rank=False)
    ric_rep = regime_ic(signal, forward_returns, regimes, rank=True)
    ret_src = strategy_returns if strategy_returns is not None else (
        np.nanmean(np.asarray(forward_returns, dtype=np.float64), axis=1)
        if np.asarray(forward_returns).ndim == 2
        else forward_returns
    )
    ret_rep = regime_returns(ret_src, regimes)
    return {
        "name": "regime_performance",
        "ic": ic_rep,
        "rank_ic": ric_rep,
        "returns": ret_rep,
    }


def regime_hit_rate(
    signal: Any,
    forward_returns: Any,
    regimes: Any,
) -> dict[str, Any]:
    """Directional hit rate by regime (sign agreement)."""
    sig = _as_1d(signal) if np.asarray(signal).ndim == 1 else np.asarray(signal, dtype=np.float64)
    ret = np.asarray(forward_returns, dtype=np.float64)
    if sig.ndim == 2:
        # collapse to mean sign agreement per date then by regime
        labels = _align_regime_labels(regimes, sig.shape[0])
        daily_hit = []
        for i in range(sig.shape[0]):
            m = np.isfinite(sig[i]) & np.isfinite(ret[i])
            if m.sum() == 0:
                daily_hit.append(np.nan)
            else:
                daily_hit.append(float(np.mean(np.sign(sig[i][m]) == np.sign(ret[i][m]))))
        hit = np.asarray(daily_hit, dtype=np.float64)
    else:
        labels = _align_regime_labels(regimes, sig.shape[0])
        hit = (np.sign(sig) == np.sign(ret.reshape(-1))).astype(np.float64)
        hit[~np.isfinite(sig) | ~np.isfinite(ret.reshape(-1))] = np.nan

    by_regime: dict[str, float] = {}
    for g in sorted({str(x) for x in labels.tolist()}, key=str):
        mask = np.asarray([str(x) == g for x in labels.tolist()])
        vals = hit[mask]
        vals = vals[np.isfinite(vals)]
        by_regime[g] = float(np.mean(vals)) if vals.size else float("nan")
    return {"name": "regime_hit_rate", "by_regime": by_regime}
