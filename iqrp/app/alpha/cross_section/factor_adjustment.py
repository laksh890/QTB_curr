"""Factor adjustment wrappers (beta / style / multi-factor)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from iqrp.app.alpha.cross_section.residualization import (
    beta_residualize,
    residualize_vs_factors,
    residualize_vs_signals,
)


def factor_neutralize(
    signal: Any,
    factor_exposures: Any,
    *,
    add_intercept: bool = True,
) -> np.ndarray:
    """Residualize signal against known risk-factor exposures."""
    return residualize_vs_factors(signal, factor_exposures, add_intercept=add_intercept)


def style_adjust(
    signal: Any,
    styles: Mapping[str, Any] | Sequence[Any],
    *,
    add_intercept: bool = True,
) -> np.ndarray:
    """Residualize against style factors (value, size, momentum, …)."""
    if isinstance(styles, Mapping):
        return residualize_vs_signals(signal, dict(styles), add_intercept=add_intercept)
    return residualize_vs_signals(signal, list(styles), add_intercept=add_intercept)


def market_beta_adjust(
    signal: Any,
    market_returns: Any,
    asset_returns: Any,
    *,
    lookback: int = 60,
) -> np.ndarray:
    """Remove market-beta component estimated from trailing returns."""
    return beta_residualize(
        signal,
        market_returns,
        asset_returns,
        lookback=lookback,
    )


def orthogonalize_to_book(
    signal: Any,
    book_signals: Mapping[str, Any] | Sequence[Any],
) -> np.ndarray:
    """Orthogonalize a candidate vs the existing alpha book."""
    return residualize_vs_signals(signal, book_signals, add_intercept=True)


def factor_exposure_summary(
    signal: Any,
    factor_exposures: Any,
) -> dict[str, Any]:
    """Cross-sectional correlation of signal with each factor (mean over time)."""
    panel = np.asarray(signal, dtype=np.float64)
    if panel.ndim == 1:
        panel = panel.reshape(1, -1)
    f = np.asarray(factor_exposures, dtype=np.float64)
    t, n = panel.shape
    if f.ndim == 2:
        if f.shape == (t, n):
            f = f.reshape(t, n, 1)
        elif f.shape[0] == n:
            f = np.broadcast_to(f.reshape(1, n, f.shape[1]), (t, n, f.shape[1])).copy()
        else:
            raise ValueError("factor_exposures shape incompatible with signal")
    k = f.shape[2]
    corrs = np.full((t, k), np.nan, dtype=np.float64)
    for i in range(t):
        s = panel[i]
        for j in range(k):
            fj = f[i, :, j]
            m = np.isfinite(s) & np.isfinite(fj)
            if m.sum() < 3:
                continue
            a, b = s[m], fj[m]
            if np.std(a) < 1e-15 or np.std(b) < 1e-15:
                continue
            corrs[i, j] = float(np.corrcoef(a, b)[0, 1])
    mean_corr = np.nanmean(corrs, axis=0)
    return {
        "name": "factor_exposure_summary",
        "n_factors": int(k),
        "mean_correlation": mean_corr.tolist(),
        "abs_mean_correlation": np.abs(mean_corr).tolist(),
        "max_abs_correlation": float(np.nanmax(np.abs(mean_corr))) if k else 0.0,
    }
