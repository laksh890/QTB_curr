"""Univariate feature statistics for research validation."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy import stats  # type: ignore[import-untyped]

from iqrp.app.features.research._numeric import shannon_entropy
from iqrp.app.features.research.config import ResearchSettings


@dataclass(frozen=True, slots=True)
class FeatureStats:
    name: str
    mean: float
    median: float
    variance: float
    std: float
    skewness: float
    kurtosis: float
    entropy: float
    missing_pct: float
    infinite_pct: float
    zero_pct: float
    unique_count: int
    distribution_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _distribution_type(x: np.ndarray, *, alpha: float) -> str:
    m = np.isfinite(x)
    if m.sum() < 20:
        return "insufficient_data"
    sample = x[m]
    # Constant
    if np.nanstd(sample) == 0:
        return "constant"
    # Uniform-ish: range covers most mass evenly
    hist, _ = np.histogram(sample, bins=10, density=True)
    if hist.std() / (hist.mean() + 1e-12) < 0.35:
        return "uniform_like"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            jb_stat, jb_p = stats.jarque_bera(sample)
        except Exception:
            jb_stat, jb_p = float("nan"), 0.0
        skew = float(stats.skew(sample))
        kurt = float(stats.kurtosis(sample))
    if np.isfinite(jb_p) and jb_p >= alpha and abs(skew) < 0.5 and abs(kurt) < 1.0:
        return "normal_like"
    if abs(skew) >= 1.0:
        return "skewed"
    if kurt >= 3.0:
        return "heavy_tailed"
    _ = jb_stat
    return "other"


def _safe_moment(sample: np.ndarray, kind: str) -> float:
    if kind == "skew" and sample.size <= 2:
        return 0.0
    if kind == "kurt" and sample.size <= 3:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "skew":
            return float(stats.skew(sample))
        return float(stats.kurtosis(sample))


class FeatureStatisticsEngine:
    """Compute descriptive statistics for each feature column."""

    def __init__(self, settings: ResearchSettings | None = None) -> None:
        self.settings = settings or ResearchSettings.default()

    def compute(self, frame: pl.DataFrame, columns: list[str]) -> list[FeatureStats]:
        results: list[FeatureStats] = []
        n = frame.height
        bins = self.settings.statistics.entropy_bins
        alpha = self.settings.statistics.jarque_bera_alpha
        for name in columns:
            s = frame[name]
            arr = s.cast(pl.Float64).to_numpy()
            finite = arr[np.isfinite(arr)]
            missing = int(s.null_count()) + int((~np.isfinite(arr) & ~np.isnan(arr)).sum())
            # Nulls already excluded from float nan; count nan separately
            nans = int(np.isnan(arr).sum()) if arr.dtype.kind == "f" else 0
            missing_pct = 100.0 * (s.null_count() + nans) / max(n, 1)
            inf_pct = 100.0 * int(np.isinf(arr).sum()) / max(n, 1)
            zero_pct = 100.0 * int((finite == 0).sum()) / max(n, 1)
            if finite.size == 0:
                results.append(
                    FeatureStats(
                        name=name,
                        mean=float("nan"),
                        median=float("nan"),
                        variance=float("nan"),
                        std=float("nan"),
                        skewness=float("nan"),
                        kurtosis=float("nan"),
                        entropy=float("nan"),
                        missing_pct=missing_pct,
                        infinite_pct=inf_pct,
                        zero_pct=zero_pct,
                        unique_count=0,
                        distribution_type="empty",
                    )
                )
                continue
            results.append(
                FeatureStats(
                    name=name,
                    mean=float(np.mean(finite)),
                    median=float(np.median(finite)),
                    variance=float(np.var(finite, ddof=1)) if finite.size > 1 else 0.0,
                    std=float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                    skewness=_safe_moment(finite, "skew"),
                    kurtosis=_safe_moment(finite, "kurt"),
                    entropy=shannon_entropy(finite, bins=bins),
                    missing_pct=missing_pct,
                    infinite_pct=inf_pct,
                    zero_pct=zero_pct,
                    unique_count=int(np.unique(finite).size),
                    distribution_type=_distribution_type(finite, alpha=alpha),
                )
            )
            _ = missing
        return results

    def to_frame(self, stats: list[FeatureStats]) -> pl.DataFrame:
        if not stats:
            return pl.DataFrame()
        return pl.DataFrame([s.to_dict() for s in stats])
