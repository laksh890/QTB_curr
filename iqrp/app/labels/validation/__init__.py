"""Label validation and quality reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.labels.base.registry import get_registry
from iqrp.app.labels.config import LabelSettings


@dataclass
class LabelQualityReport:
    name: str
    class_distribution: dict[str, float]
    entropy: float
    information_content: float
    prediction_horizon: int
    coverage: float
    missing_pct: float
    n_unique: int
    is_degenerate: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LabelValidationReport:
    look_ahead_flags: list[str] = field(default_factory=list)
    leakage_flags: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    duplicate_labels: list[tuple[str, str]] = field(default_factory=list)
    class_imbalance: list[str] = field(default_factory=list)
    degenerate_labels: list[str] = field(default_factory=list)
    quality: list[LabelQualityReport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "look_ahead_flags": self.look_ahead_flags,
            "leakage_flags": self.leakage_flags,
            "missing_labels": self.missing_labels,
            "duplicate_labels": [list(x) for x in self.duplicate_labels],
            "class_imbalance": self.class_imbalance,
            "degenerate_labels": self.degenerate_labels,
            "quality": [q.to_dict() for q in self.quality],
        }


class LabelValidator:
    def __init__(self, settings: LabelSettings | None = None) -> None:
        self.settings = settings or LabelSettings.default()

    def validate(
        self,
        frame: pl.DataFrame,
        label_columns: list[str] | None = None,
    ) -> LabelValidationReport:
        reg = get_registry()
        cols = label_columns or [
            c
            for c in frame.columns
            if c not in {"open_time", "open", "high", "low", "close", "volume"}
            and frame[c].dtype.is_numeric()
        ]
        report = LabelValidationReport()
        arrays = {c: frame[c].cast(pl.Float64).to_numpy() for c in cols}

        # Missing / degenerate / imbalance / quality
        for c in cols:
            arr = arrays[c]
            miss = float(np.mean(~np.isfinite(arr))) if len(arr) else 1.0
            if miss >= 1.0:
                report.missing_labels.append(c)
            finite = arr[np.isfinite(arr)]
            n_unique = int(np.unique(finite).size) if finite.size else 0
            degenerate = n_unique <= 1
            if degenerate:
                report.degenerate_labels.append(c)

            dist: dict[str, float] = {}
            if finite.size and n_unique <= 20:
                vals, counts = np.unique(finite, return_counts=True)
                total = counts.sum()
                for v, cnt in zip(vals, counts, strict=False):
                    dist[str(v)] = float(cnt / total)
                if dist:
                    mn = min(dist.values())
                    if mn < self.settings.validation.imbalance_ratio_alert:
                        report.class_imbalance.append(c)

            ent = _entropy(finite, self.settings.validation.entropy_bins)
            coverage = 1.0 - miss
            if coverage < self.settings.validation.min_coverage and c not in report.missing_labels:
                report.missing_labels.append(c)

            horizon = 0
            try:
                # Map output column back to registered label when possible
                for name in reg.list_names():
                    meta = reg.describe(name)
                    if c in meta.output_columns or c == name:
                        horizon = meta.prediction_horizon
                        # Soft look-ahead flag when trailing horizon rows are all finite.
                        if (
                            horizon > 0
                            and len(arr) > horizon
                            and np.all(np.isfinite(arr[-horizon:]))
                        ):
                            report.look_ahead_flags.append(c)
                        break
            except Exception:
                horizon = 0

            report.quality.append(
                LabelQualityReport(
                    name=c,
                    class_distribution=dist,
                    entropy=ent,
                    information_content=ent,  # Shannon entropy as IC proxy for discrete labels
                    prediction_horizon=horizon,
                    coverage=coverage,
                    missing_pct=100.0 * miss,
                    n_unique=n_unique,
                    is_degenerate=degenerate,
                )
            )

        # Duplicate label columns
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                if np.allclose(
                    np.nan_to_num(arrays[a], nan=0.0),
                    np.nan_to_num(arrays[b], nan=0.0),
                    equal_nan=True,
                ):
                    report.duplicate_labels.append((a, b))

        # Leakage heuristic: label correlates perfectly with contemporaneous close changes
        if "close" in frame.columns:
            close_ret = frame.select(pl.col("close").pct_change().alias("r")).to_series().to_numpy()
            for c in cols:
                m = np.isfinite(arrays[c]) & np.isfinite(close_ret)
                if m.sum() < 30:
                    continue
                if np.std(arrays[c][m]) == 0 or np.std(close_ret[m]) == 0:
                    continue
                corr = float(np.corrcoef(arrays[c][m], close_ret[m])[0, 1])
                if abs(corr) > 0.999:
                    report.leakage_flags.append(c)

        return report


def _entropy(values: np.ndarray, bins: int) -> float:
    if values.size < 2:
        return float("nan")
    uniq = np.unique(values)
    if uniq.size <= 20:
        _, counts = np.unique(values, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p + 1e-12)).sum())
    hist, _ = np.histogram(values, bins=bins)
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())
