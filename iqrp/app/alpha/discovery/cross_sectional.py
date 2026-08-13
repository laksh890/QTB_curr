"""Cross-sectional discovery templates.

CRITICAL:
- Point-in-time: cross-sectional ranks at time t use only contemporaneous
  feature values known at t (no future leakage across time).
- Statistical significance alone ≠ alpha.
- Must track economic_hypothesis on SignalDefinition.
- Templates do not claim profitability.
"""

from __future__ import annotations

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from iqrp.app.alpha.base.alpha_signal import AlphaSignal
from iqrp.app.alpha.base.signal_definition import SignalDefinition


def _cs_rank_matrix(x: np.ndarray) -> np.ndarray:
    """Rank each row cross-sectionally to [0, 1]. NaNs preserved."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    out = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        row = x[i]
        m = np.isfinite(row)
        if m.sum() == 0:
            continue
        ranks = stats.rankdata(row[m], method="average")
        out[i, m] = (ranks - 1.0) / max(m.sum() - 1, 1)
    return out


def cross_sectional_rank_signal(
    feature_panel: np.ndarray,
    *,
    asset_index: int = 0,
    name: str = "cs_rank",
    owner: str = "research",
    universe: str = "default",
    frequency: str = "1d",
    lookback: int = 1,
    horizon: int = 1,
    economic_hypothesis: str | None = None,
) -> AlphaSignal:
    """Extract one asset's cross-sectional rank through time.

    ``feature_panel`` shape: (T, N) — rows are dates, columns are assets.
    At each t, ranks use only the row at t (no lead of future rows).
    """
    panel = np.asarray(feature_panel, dtype=np.float64)
    if panel.ndim != 2:
        raise ValueError(f"feature_panel must be 2-D (T, N), got {panel.shape}")
    if asset_index < 0 or asset_index >= panel.shape[1]:
        raise ValueError(f"asset_index {asset_index} out of range for N={panel.shape[1]}")
    ranks = _cs_rank_matrix(panel)
    values = ranks[:, asset_index]
    hyp = economic_hypothesis or (
        "Relative valuation / characteristic ranking within a universe can reflect "
        "compensated risk or behavioral mispricing differentials; CS rank is a "
        "research candidate without a profitability claim."
    )
    definition = SignalDefinition(
        name=name,
        version="1.0.0",
        formula=f"cs_rank(feature)[:, {asset_index}]",
        features=("feature_panel",),
        lookback=lookback,
        horizon=horizon,
        universe=universe,
        frequency=frequency,
        direction="long_short",
        expected_relationship="unknown",
        economic_hypothesis=hyp,
        owner=owner,
        signal_type="cross_sectional",
        parameters={"asset_index": asset_index, "n_assets": int(panel.shape[1])},
        tags=("cross_sectional", "rank", "candidate"),
    )
    return AlphaSignal(
        values=values,
        name=definition.name,
        definition_id=definition.definition_id,
        metadata={
            "definition": definition.to_dict(),
            "template": "cross_sectional_rank",
            "claims_profitability": False,
        },
    )


def cross_sectional_zscore_signal(
    feature_panel: np.ndarray,
    *,
    asset_index: int = 0,
    name: str = "cs_zscore",
    owner: str = "research",
    universe: str = "default",
    frequency: str = "1d",
    lookback: int = 1,
    horizon: int = 1,
    economic_hypothesis: str | None = None,
) -> AlphaSignal:
    """Cross-sectional z-score of one asset through time (PIT per row)."""
    panel = np.asarray(feature_panel, dtype=np.float64)
    if panel.ndim != 2:
        raise ValueError(f"feature_panel must be 2-D (T, N), got {panel.shape}")
    if asset_index < 0 or asset_index >= panel.shape[1]:
        raise ValueError(f"asset_index {asset_index} out of range")
    out = np.full(panel.shape[0], np.nan, dtype=np.float64)
    for t in range(panel.shape[0]):
        row = panel[t]
        m = np.isfinite(row)
        if m.sum() < 3:
            continue
        mu = float(np.mean(row[m]))
        sd = float(np.std(row[m], ddof=1))
        if sd <= 1e-12 or not np.isfinite(panel[t, asset_index]):
            continue
        out[t] = (panel[t, asset_index] - mu) / sd
    hyp = economic_hypothesis or (
        "Cross-sectional standardization isolates relative characteristic extremes "
        "that may be linked to risk premia or temporary dislocations; candidate only."
    )
    definition = SignalDefinition(
        name=name,
        version="1.0.0",
        formula=f"cs_zscore(feature)[:, {asset_index}]",
        features=("feature_panel",),
        lookback=lookback,
        horizon=horizon,
        universe=universe,
        frequency=frequency,
        direction="long_short",
        expected_relationship="unknown",
        economic_hypothesis=hyp,
        owner=owner,
        signal_type="cross_sectional",
        parameters={"asset_index": asset_index},
        tags=("cross_sectional", "zscore", "candidate"),
    )
    return AlphaSignal(
        values=out,
        name=definition.name,
        definition_id=definition.definition_id,
        metadata={
            "definition": definition.to_dict(),
            "template": "cross_sectional_zscore",
            "claims_profitability": False,
        },
    )


def long_short_spread(
    feature_panel: np.ndarray,
    *,
    top_frac: float = 0.2,
    bottom_frac: float = 0.2,
) -> np.ndarray:
    """Per-date long-short spread of feature (diagnostic, not a position).

    Returns shape (T,) equal-weight mean(top) - mean(bottom) of the feature.
    Point-in-time: each date uses only that date's cross-section.
    """
    panel = np.asarray(feature_panel, dtype=np.float64)
    if panel.ndim != 2:
        raise ValueError("feature_panel must be 2-D")
    if not (0 < top_frac < 1 and 0 < bottom_frac < 1):
        raise ValueError("fractions must be in (0, 1)")
    out = np.full(panel.shape[0], np.nan, dtype=np.float64)
    for t in range(panel.shape[0]):
        row = panel[t]
        m = np.isfinite(row)
        vals = row[m]
        n = vals.size
        if n < 4:
            continue
        k_top = max(1, int(np.floor(n * top_frac)))
        k_bot = max(1, int(np.floor(n * bottom_frac)))
        order = np.argsort(vals)
        bot = vals[order[:k_bot]]
        top = vals[order[-k_top:]]
        out[t] = float(np.mean(top) - np.mean(bot))
    return out
