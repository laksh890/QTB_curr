"""Cross-sectional ranking, neutralization, and factor adjustment."""

from __future__ import annotations

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

__all__ = [
    "beta_residualize",
    "cap_weighted_sector_neutral",
    "cross_sectional_minmax",
    "cross_sectional_percentile",
    "cross_sectional_rank",
    "cross_sectional_zscore",
    "demean_by_group",
    "factor_exposure_summary",
    "factor_neutralize",
    "industry_neutralize",
    "market_beta_adjust",
    "neutralize_market",
    "neutralize_multi_group",
    "neutralize_weighted",
    "orthogonalize_to_book",
    "residualize_vs_factors",
    "residualize_vs_signals",
    "sector_neutral_zscore",
    "sector_relative_ranks",
    "style_adjust",
    "winsorize_cross_section",
]
