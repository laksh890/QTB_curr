"""Redundancy, multicollinearity, and removal suggestions."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.features.research._numeric import pearson
from iqrp.app.features.research.config import ResearchSettings


@dataclass
class RedundancyReport:
    duplicates: list[tuple[str, str]]
    near_duplicates: list[tuple[str, str, float]]
    linear_dependence_rank: int
    feature_count: int
    vif: dict[str, float]
    multicollinear_features: list[str]
    redundant_rolling_windows: list[list[str]]
    suggested_removals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duplicates"] = [list(x) for x in self.duplicates]
        d["near_duplicates"] = [{"a": a, "b": b, "corr": c} for a, b, c in self.near_duplicates]
        return d


class RedundancyDetector:
    def __init__(self, settings: ResearchSettings | None = None) -> None:
        self.settings = settings or ResearchSettings.default()

    def detect(self, frame: pl.DataFrame, columns: list[str]) -> RedundancyReport:
        cfg = self.settings.redundancy
        if not columns:
            return RedundancyReport(
                duplicates=[],
                near_duplicates=[],
                linear_dependence_rank=0,
                feature_count=0,
                vif={},
                multicollinear_features=[],
                redundant_rolling_windows=[],
            )

        arrays = {c: frame[c].cast(pl.Float64).to_numpy() for c in columns}
        dups: list[tuple[str, str]] = []
        near: list[tuple[str, str, float]] = []
        for i, a in enumerate(columns):
            for b in columns[i + 1 :]:
                xa, xb = arrays[a], arrays[b]
                if xa.shape == xb.shape and np.allclose(
                    np.nan_to_num(xa, nan=0.0),
                    np.nan_to_num(xb, nan=0.0),
                    atol=cfg.duplicate_atol,
                    equal_nan=True,
                ):
                    dups.append((a, b))
                r = pearson(xa, xb)
                if np.isfinite(r) and abs(r) >= cfg.near_duplicate_threshold:
                    near.append((a, b, float(r)))

        mat = np.column_stack([np.nan_to_num(arrays[c], nan=0.0) for c in columns])
        rank = int(np.linalg.matrix_rank(mat, tol=cfg.linear_dependence_rank_tol))

        vif: dict[str, float] = {}
        for j, name in enumerate(columns):
            y = mat[:, j]
            x = np.delete(mat, j, axis=1)
            if x.shape[1] == 0:
                vif[name] = 1.0
                continue
            # R^2 from least squares
            try:
                coef, *_ = np.linalg.lstsq(x, y, rcond=None)
                pred = x @ coef
                ss_res = float(np.sum((y - pred) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                r2 = min(max(r2, 0.0), 0.999999)
                vif[name] = float(1.0 / (1.0 - r2))
            except Exception:
                vif[name] = float("inf")

        multi = [n for n, v in vif.items() if np.isfinite(v) and v >= cfg.vif_threshold]

        # Group features that look like same family with different windows
        pattern = re.compile(cfg.rolling_window_name_pattern, re.IGNORECASE)
        families: dict[str, list[str]] = {}
        for name in columns:
            base = pattern.sub("", name)
            families.setdefault(base, []).append(name)
        roll_groups = [sorted(v) for v in families.values() if len(v) > 1]

        suggested: list[str] = []
        for _a, b in dups:
            if b not in suggested:
                suggested.append(b)
        for _a, b, _corr in near:
            if b not in suggested and b not in [x for x, _ in dups]:
                suggested.append(b)
        for name in multi:
            if name not in suggested:
                suggested.append(name)
        for group in roll_groups:
            # keep first, suggest rest
            for name in group[1:]:
                if name not in suggested:
                    suggested.append(name)

        return RedundancyReport(
            duplicates=dups,
            near_duplicates=near,
            linear_dependence_rank=rank,
            feature_count=len(columns),
            vif=vif,
            multicollinear_features=multi,
            redundant_rolling_windows=roll_groups,
            suggested_removals=suggested,
        )
