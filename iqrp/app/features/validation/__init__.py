"""Feature matrix validation utilities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class FeatureValidationReport:
    nan_counts: dict[str, int]
    inf_counts: dict[str, int]
    duplicate_feature_pairs: tuple[tuple[str, str], ...]
    constant_features: tuple[str, ...]
    low_variance_features: tuple[str, ...]
    highly_correlated_pairs: tuple[tuple[str, str, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "nan_counts": self.nan_counts,
            "inf_counts": self.inf_counts,
            "duplicate_feature_pairs": list(self.duplicate_feature_pairs),
            "constant_features": list(self.constant_features),
            "low_variance_features": list(self.low_variance_features),
            "highly_correlated_pairs": [
                {"a": a, "b": b, "corr": c} for a, b, c in self.highly_correlated_pairs
            ],
        }


class FeatureValidator:
    """Detect NaN/Inf, duplicates, constants, low variance, and high correlation."""

    def __init__(
        self,
        *,
        variance_epsilon: float = 1e-12,
        corr_threshold: float = 0.99,
    ) -> None:
        self.variance_epsilon = variance_epsilon
        self.corr_threshold = corr_threshold

    def validate(
        self,
        frame: pl.DataFrame,
        columns: list[str] | None = None,
    ) -> FeatureValidationReport:
        cols = columns or [
            c
            for c, dt in zip(frame.columns, frame.dtypes, strict=False)
            if dt.is_numeric() and c != "open_time"
        ]
        nan_counts: dict[str, int] = {}
        inf_counts: dict[str, int] = {}
        constants: list[str] = []
        low_var: list[str] = []

        for c in cols:
            s = frame[c]
            nan_counts[c] = int(s.null_count())
            # Inf detection via casting comparison
            try:
                inf_counts[c] = int(frame.select(pl.col(c).is_infinite().sum()).item())
            except Exception:
                inf_counts[c] = 0
            variance = frame.select(pl.col(c).var()).item()
            if variance is None or variance == 0:
                constants.append(c)
            elif float(variance) < self.variance_epsilon:
                low_var.append(c)

        # Exact duplicate columns
        dup_pairs: list[tuple[str, str]] = []
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                if frame[a].equals(frame[b]):
                    dup_pairs.append((a, b))

        # High correlation (skip constants / low variance; ignore NaN corr coeffs)
        high_corr: list[tuple[str, str, float]] = []
        varying = [c for c in cols if c not in constants and c not in low_var]
        if len(varying) >= 2:
            sub = frame.select(varying).drop_nulls()
            if sub.height >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    corr = sub.corr()
                names = corr.columns
                mat = corr.to_numpy()
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        val = float(mat[i, j])
                        if val == val and abs(val) >= self.corr_threshold:  # not NaN
                            high_corr.append((names[i], names[j], val))

        return FeatureValidationReport(
            nan_counts=nan_counts,
            inf_counts=inf_counts,
            duplicate_feature_pairs=tuple(dup_pairs),
            constant_features=tuple(constants),
            low_variance_features=tuple(low_var),
            highly_correlated_pairs=tuple(high_corr),
        )
